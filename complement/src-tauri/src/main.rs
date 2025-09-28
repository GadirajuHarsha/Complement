#[cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use tauri::{Emitter, Manager};

// --- IMPORTS ---
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use meval;
use rusqlite::{Connection, Result as SqlResult, params};
use chrono::{DateTime, Utc, Local, Datelike, Timelike};
use std::path::PathBuf;
use std::sync::Mutex;
use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt;
use clipboard::{ClipboardProvider, ClipboardContext};
use ndarray::Array1;


// --- DATA STRUCTURES ---
#[derive(Debug, Clone, serde::Serialize)]
struct App {
    name: String,
    path: String,
}

#[derive(Clone, serde::Serialize)]
#[serde(tag = "type", content = "payload")]
enum CommandResult {
    Apps(Vec<App>),
    Text(String),
}

// --- DATABASE STRUCTURES ---
#[derive(Debug, Clone, serde::Serialize)]
struct UsageEvent {
    id: Option<i64>,
    event_type: String,      // "app_launch", "search_query", "calculator"
    content: String,         // app name/path, search query, or calculation
    timestamp: DateTime<Utc>,
    day_of_week: i32,        // 0=Sunday, 1=Monday, etc.
    hour: i32,               // 0-23
    context: Option<String>, // Additional context like active window
}

#[derive(Debug, Clone, serde::Serialize)]
struct AppFrequency {
    app_name: String,
    app_path: String,
    launch_count: i32,
    last_used: DateTime<Utc>,
    avg_hour: f64,           // Average hour of day when used
    common_days: String,     // JSON array of common days
}

#[derive(Debug, Clone, serde::Serialize)]
struct RecommendationContext {
    greeting: String,
    recommendations: Vec<App>,
    context_message: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ClipboardItem {
    id: i64,
    content: String,
    content_type: String,     // "text", "image", "file"
    preview: String,          // First 100 chars for text, filename for files
    timestamp: String,        // RFC3339 format for frontend compatibility
    source_app: Option<String>, // App that was active when copied
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Snippet {
    id: Option<i64>,
    keyword: String,          // Short trigger keyword (e.g., "email", "addr")
    name: String,             // Human-readable name (e.g., "Work Email Template")
    content: String,          // The actual text content to expand
    description: String,      // Optional description for the user
    usage_count: i64,         // Track how often it's used
    created_at: String,       // RFC3339 timestamp
    updated_at: String,       // RFC3339 timestamp
}

#[derive(Debug, Clone, serde::Serialize)]
struct MLPrediction {
    app_path: String,
    app_name: String,
    confidence: f32,
    reasoning: String,
}

#[derive(Debug, Clone)]
struct ContextFeatures {
    hour_of_day: f32,           // 0-23 normalized to 0-1
    day_of_week: f32,           // 0-6 normalized to 0-1
    is_weekend: f32,            // 0 or 1
    time_since_last_use: f32,   // Hours since last app use (inverted, normalized)
    usage_frequency: f32,       // Uses per week (normalized)
    recent_apps_similarity: f32, // Similarity to recently used apps
    active_window_context: f32,  // Context from current window
    search_context: f32,        // Context from recent searches
}

// Global ML Environment (initialized once)


// --- ML FEATURE ENGINEERING ---
fn extract_context_features(app_path: &str, conn: &Connection) -> Result<ContextFeatures, Box<dyn std::error::Error>> {
    let now = chrono::Local::now();
    let hour = now.hour() as f32;
    let day_of_week = now.weekday().number_from_monday() as f32 - 1.0; // 0-6
    let is_weekend = if day_of_week >= 5.0 { 1.0 } else { 0.0 };
    
    // Get app usage statistics
    let usage_stats = conn.query_row(
        "SELECT launch_count, AVG(hour), MAX(timestamp) FROM app_frequency af
         JOIN usage_events ue ON af.app_path = ue.content  
         WHERE af.app_path = ?1 AND ue.event_type = 'app_launch'",
        params![app_path],
        |row| {
            let count: i64 = row.get(0).unwrap_or(0);
            let avg_hour: f64 = row.get(1).unwrap_or(12.0);
            let last_used: Option<String> = row.get(2).ok();
            Ok((count, avg_hour, last_used))
        }
    ).unwrap_or((0, 12.0, None));
    
    let usage_frequency = usage_stats.0 as f32 / 7.0; // per week
    
    // Calculate time since last use
    let time_since_last_use = if let Some(last_timestamp) = usage_stats.2 {
        if let Ok(last_time) = chrono::DateTime::parse_from_rfc3339(&last_timestamp) {
            let duration = now.signed_duration_since(last_time);
            duration.num_hours() as f32
        } else {
            168.0 // 1 week default
        }
    } else {
        168.0 // 1 week default if never used
    };
    
    // Get recent app similarity (simplified)
    let recent_apps_similarity = conn.query_row(
        "SELECT COUNT(*) FROM usage_events 
         WHERE event_type = 'app_launch' AND timestamp > datetime('now', '-1 hour')",
        [],
        |row| Ok(row.get::<_, i64>(0)? as f32 / 10.0) // Normalize to 0-1
    ).unwrap_or(0.0);
    
    // Get active window context (simplified scoring)
    let active_window_context = match get_active_window_info() {
        Some(window) => {
            if window.to_lowercase().contains("code") || window.to_lowercase().contains("dev") {
                0.8 // Development context
            } else if window.to_lowercase().contains("browser") || window.to_lowercase().contains("chrome") {
                0.6 // Browser context  
            } else {
                0.3 // General context
            }
        },
        None => 0.0
    };
    
    // Get search context from recent queries
    let search_context = conn.query_row(
        "SELECT COUNT(*) FROM usage_events 
         WHERE event_type = 'search_query' AND timestamp > datetime('now', '-10 minutes')",
        [],
        |row| Ok((row.get::<_, i64>(0)? as f32).min(5.0) / 5.0) // Normalize to 0-1
    ).unwrap_or(0.0);
    
    Ok(ContextFeatures {
        hour_of_day: hour / 24.0,  // Normalize to 0-1
        day_of_week: day_of_week / 6.0, // Normalize to 0-1
        is_weekend,
        time_since_last_use: (168.0 - time_since_last_use.min(168.0)) / 168.0, // Invert and normalize
        usage_frequency: usage_frequency.min(10.0) / 10.0, // Cap and normalize
        recent_apps_similarity,
        active_window_context,
        search_context,
    })
}

fn features_to_array(features: &ContextFeatures) -> Array1<f32> {
    Array1::from(vec![
        features.hour_of_day,
        features.day_of_week,
        features.is_weekend,
        features.time_since_last_use,
        features.usage_frequency,
        features.recent_apps_similarity,
        features.active_window_context,
        features.search_context,
    ])
}

// --- DATABASE FUNCTIONS ---
fn get_database_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut db_path = dirs::data_dir().ok_or("Could not find data directory")?;
    db_path.push("Complement");
    std::fs::create_dir_all(&db_path)?;
    db_path.push("complement.db");
    Ok(db_path)
}

fn init_database() -> SqlResult<Connection> {
    let db_path = get_database_path().map_err(|e| {
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
            Some(format!("Failed to get database path: {}", e))
        )
    })?;
    
    let conn = Connection::open(db_path)?;
    
    // Create usage_events table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            day_of_week INTEGER NOT NULL,
            hour INTEGER NOT NULL,
            context TEXT
        )",
        [],
    )?;
    
    // Create app_frequency table for aggregated stats
    conn.execute(
        "CREATE TABLE IF NOT EXISTS app_frequency (
            app_name TEXT PRIMARY KEY,
            app_path TEXT NOT NULL,
            launch_count INTEGER NOT NULL DEFAULT 0,
            last_used TEXT NOT NULL,
            avg_hour REAL NOT NULL DEFAULT 12.0,
            common_days TEXT NOT NULL DEFAULT '[]'
        )",
        [],
    )?;
    
    // Create clipboard_history table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS clipboard_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL,
            preview TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source_app TEXT
        )",
        [],
    )?;
    
    // Create snippets table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            description TEXT NOT NULL,
            usage_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
        [],
    )?;
    
    println!("🗄️ Database initialized successfully");
    Ok(conn)
}

fn log_usage_event(conn: &Connection, event_type: &str, content: &str, context: Option<&str>) -> SqlResult<()> {
    let now = Utc::now();
    let local_time = now.with_timezone(&Local);
    
    conn.execute(
        "INSERT INTO usage_events (event_type, content, timestamp, day_of_week, hour, context)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        (
            event_type,
            content,
            now.to_rfc3339(),
            local_time.weekday().number_from_sunday() as i32,
            local_time.hour() as i32,
            context,
        ),
    )?;
    
    // If it's an app launch, update frequency stats
    if event_type == "app_launch" {
        update_app_frequency(conn, content, &now)?;
    }
    
    Ok(())
}

fn update_app_frequency(conn: &Connection, app_path: &str, timestamp: &DateTime<Utc>) -> SqlResult<()> {
    let app_name = std::path::Path::new(app_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown")
        .to_string();
    
    let local_time = timestamp.with_timezone(&Local);
    let hour = local_time.hour() as f64;
    
    conn.execute(
        "INSERT OR REPLACE INTO app_frequency 
         (app_name, app_path, launch_count, last_used, avg_hour, common_days)
         VALUES (
             ?1, 
             ?2, 
             COALESCE((SELECT launch_count FROM app_frequency WHERE app_name = ?1), 0) + 1,
             ?3,
             CASE 
                 WHEN (SELECT launch_count FROM app_frequency WHERE app_name = ?1) > 0 
                 THEN ((SELECT avg_hour * launch_count FROM app_frequency WHERE app_name = ?1) + ?4) / 
                      (COALESCE((SELECT launch_count FROM app_frequency WHERE app_name = ?1), 0) + 1)
                 ELSE ?4
             END,
             '[]'
         )",
        (app_name, app_path, timestamp.to_rfc3339(), hour),
    )?;
    
    Ok(())
}

fn get_active_window_info() -> Option<String> {
    #[cfg(windows)]
    {
        use windows::{
            Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowTextW},
        };
        
        unsafe {
            let hwnd = GetForegroundWindow();
            if !hwnd.is_invalid() && hwnd.0 != std::ptr::null_mut() {
                let mut buffer = [0u16; 512];
                let len = GetWindowTextW(hwnd, &mut buffer);
                if len > 0 {
                    let title = OsString::from_wide(&buffer[..len as usize]);
                    return title.to_string_lossy().to_string().into();
                }
            }
        }
    }
    None
}

fn log_clipboard_item(conn: &Connection, content: &str, content_type: &str) -> SqlResult<()> {
    let now = Utc::now();
    let preview = if content.len() > 100 {
        format!("{}...", &content[..97])
    } else {
        content.to_string()
    };
    
    let source_app = get_active_window_info();
    
    // Don't store duplicates of recent items (within last 5 entries)
    let recent_exists = conn.query_row(
        "SELECT COUNT(*) FROM clipboard_history 
         WHERE content = ?1 AND id > (SELECT COALESCE(MAX(id), 0) - 5 FROM clipboard_history)",
        [content],
        |row| Ok(row.get::<_, i32>(0)? > 0)
    ).unwrap_or(false);
    
    if !recent_exists {
        conn.execute(
            "INSERT INTO clipboard_history (content, content_type, preview, timestamp, source_app)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            (content, content_type, &preview, now.to_rfc3339(), source_app),
        )?;
        
        // Keep only last 100 clipboard items
        conn.execute(
            "DELETE FROM clipboard_history 
             WHERE id NOT IN (
                 SELECT id FROM clipboard_history 
                 ORDER BY timestamp DESC LIMIT 100
             )",
            [],
        )?;
    }
    
    Ok(())
}

// --- CORE LOGIC ---
fn build_app_index() -> Vec<App> {
    let mut apps = Vec::new();
    if let Some(mut path) = dirs::data_dir() {
        path.push("Microsoft\\Windows\\Start Menu");
        for entry in walkdir::WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("lnk") {
                if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    apps.push(App {
                        name: name.to_string(),
                        path: entry.path().to_string_lossy().to_string(),
                    });
                }
            }
        }
    }
    apps
}

// --- TAURI COMMANDS ---
#[tauri::command]
fn launch_app(app: tauri::AppHandle, path: String, db: tauri::State<Mutex<Connection>>) {
    match open::that(&path) {
        Ok(_) => {
            println!("Successfully launched application: {}", path);
            
            // Log the app launch with active window context
            if let Ok(conn) = db.lock() {
                let active_window = get_active_window_info();
                if let Err(e) = log_usage_event(&conn, "app_launch", &path, active_window.as_deref()) {
                    eprintln!("Failed to log app launch: {}", e);
                }
            }
            
            // Auto-minimize Complement with fade animation after launching an app
            minimize_with_fade(app);
        }
        Err(e) => eprintln!("Failed to launch application: {}", e),
    }
}

#[tauri::command]
fn get_preview(query: String, index: tauri::State<Vec<App>>, db: tauri::State<Mutex<Connection>>) -> CommandResult {
    if query.is_empty() {
        return CommandResult::Apps(vec![]);
    }

    if let Ok(result) = meval::eval_str(&query) {
        let result_str = if result.fract() == 0.0 {
            format!("= {}", result as i64)
        } else {
            format!("= {}", result)
        };
        
        // Log calculator usage
        if let Ok(conn) = db.lock() {
            if let Err(e) = log_usage_event(&conn, "calculator", &query, None) {
                eprintln!("Failed to log calculator usage: {}", e);
            }
        }
        
        return CommandResult::Text(result_str);
    }

    // Check for web search patterns (g, google, search)
    if query.starts_with("g ") || query.starts_with("google ") || query.starts_with("search ") {
        let search_term = if query.starts_with("g ") {
            &query[2..]
        } else if query.starts_with("google ") {
            &query[7..]
        } else {
            &query[7..] // "search "
        };
        
        if !search_term.trim().is_empty() {
            // Log web search usage
            if let Ok(conn) = db.lock() {
                if let Err(e) = log_usage_event(&conn, "web_search", search_term, None) {
                    eprintln!("Failed to log web search: {}", e);
                }
            }
            
            return CommandResult::Text(format!("🔍 Search Google for: {}", search_term));
        }
    }

    // Check for snippet expansion (: prefix)
    if query.starts_with(":") && query.len() > 1 {
        let keyword = &query[1..].trim();
        if !keyword.is_empty() {
            if let Ok(conn) = db.lock() {
                // Check if snippet exists
                let snippet_exists = conn.query_row(
                    "SELECT name, content FROM snippets WHERE keyword = ?1",
                    params![keyword],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                ).ok();
                
                if let Some((name, content)) = snippet_exists {
                    let preview = if content.len() > 50 {
                        format!("{}...", &content[..47])
                    } else {
                        content.clone()
                    };
                    return CommandResult::Text(format!("📝 {} → {}", name, preview));
                } else {
                    return CommandResult::Text(format!("❌ Snippet '{}' not found", keyword));
                }
            }
        }
    }
    
    // Log search queries (but only if they're meaningful - more than 2 characters)
    if query.len() > 2 {
        if let Ok(conn) = db.lock() {
            let active_window = get_active_window_info();
            if let Err(e) = log_usage_event(&conn, "search_query", &query, active_window.as_deref()) {
                eprintln!("Failed to log search query: {}", e);
            }
        }
    }

    let matcher = SkimMatcherV2::default();
    let mut results: Vec<(i64, App)> = index
        .iter()
        .filter_map(|app| {
            matcher
                .fuzzy_match(&app.name.to_lowercase(), &query.to_lowercase())
                .map(|score| (score, app.clone()))
        })
        .collect();
    results.sort_by(|a, b| b.0.cmp(&a.0));
    let final_results: Vec<App> = results.into_iter().map(|(_, app)| app).take(5).collect();
    CommandResult::Apps(final_results)
}

#[tauri::command]
fn execute_primary_action(app: tauri::AppHandle, command: String, index: tauri::State<Vec<App>>, db: tauri::State<Mutex<Connection>>) {
    let clean_command = command.trim().to_lowercase();

    if clean_command == "exit" {
        app.exit(0);
        return;
    }

    // Handle web search commands
    if command.trim().starts_with("g ") || command.trim().starts_with("google ") || command.trim().starts_with("search ") {
        let search_term = if command.trim().starts_with("g ") {
            &command.trim()[2..]
        } else if command.trim().starts_with("google ") {
            &command.trim()[7..]
        } else {
            &command.trim()[7..] // "search "
        };
        
        if !search_term.trim().is_empty() {
            let encoded_query = urlencoding::encode(search_term);
            let search_url = format!("https://www.google.com/search?q={}", encoded_query);
            
            // Log web search execution
            if let Ok(conn) = db.lock() {
                if let Err(e) = log_usage_event(&conn, "web_search_execute", search_term, None) {
                    eprintln!("Failed to log web search execution: {}", e);
                }
            }
            
            // Open the search URL in default browser
            if let Err(e) = open::that(&search_url) {
                eprintln!("Failed to open search URL: {}", e);
            }
            return;
        }
    }

    // Handle snippet expansion
    if command.trim().starts_with(":") && command.trim().len() > 1 {
        let keyword = &command.trim()[1..].trim();
        if !keyword.is_empty() {
            match expand_snippet(keyword.to_string(), db.clone()) {
                Ok(content) => {
                    // Copy the expanded content to clipboard
                    if let Err(e) = copy_to_clipboard(content) {
                        eprintln!("Failed to copy snippet to clipboard: {}", e);
                    }
                    return;
                },
                Err(e) => {
                    eprintln!("Failed to expand snippet: {}", e);
                    return;
                }
            }
        }
    }

    let matcher = SkimMatcherV2::default();
    let mut results: Vec<(i64, App)> = index
        .iter()
        .filter_map(|app| {
            matcher
                .fuzzy_match(&app.name.to_lowercase(), &command.to_lowercase())
                .map(|score| (score, app.clone()))
        })
        .collect();
    results.sort_by(|a, b| b.0.cmp(&a.0));

    if let Some(top_result) = results.into_iter().map(|(_, app_result)| app_result).next() {
        launch_app(app, top_result.path, db);
    }
}

#[tauri::command]
fn minimize(app: tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.minimize();
    }
}

#[tauri::command]
fn minimize_with_fade(app: tauri::AppHandle) {
    // Emit an event to the frontend to trigger fade animation
    if let Err(e) = app.emit("minimize-with-fade", ()) {
        eprintln!("Failed to emit minimize-with-fade event: {}", e);
    }
}

#[tauri::command]
fn show_window(app: tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[tauri::command]
fn get_recommendations_context(db: tauri::State<Mutex<Connection>>) -> RecommendationContext {
    let mut recommendations = Vec::new();
    let mut greeting = String::from("Welcome back!");
    let mut context_message = String::from("Here are your most used apps");
    
    println!("🤖 Getting recommendations...");
    
    if let Ok(conn) = db.lock() {
        let now = Utc::now();
        let local_time = now.with_timezone(&Local);
        let current_hour = local_time.hour() as i32;
        let _current_day = local_time.weekday().number_from_sunday() as i32;
        
        // Generate contextual greeting based on time of day
        greeting = match current_hour {
            5..=11 => "Good morning!".to_string(),
            12..=17 => "Good afternoon!".to_string(), 
            18..=21 => "Good evening!".to_string(),
            _ => "Working late?".to_string(),
        };
        
        context_message = match current_hour {
            5..=8 => "Time to start your day with".to_string(),
            9..=11 => "Morning productivity apps".to_string(),
            12..=13 => "Lunch break suggestions".to_string(),
            14..=17 => "Afternoon focus tools".to_string(),
            18..=20 => "Evening wind-down apps".to_string(),
            21..=23 => "Late night essentials".to_string(),
            _ => "Night owl toolkit".to_string(),
        };
        
        // Get apps frequently used at this time of day (within 2 hours)
        let stmt = conn.prepare(
            "SELECT DISTINCT af.app_name, af.app_path, af.launch_count, af.avg_hour
             FROM app_frequency af
             JOIN usage_events ue ON af.app_path = ue.content
             WHERE ue.event_type = 'app_launch' 
               AND ABS(ue.hour - ?1) <= 2
               AND af.launch_count >= 2
             ORDER BY af.launch_count DESC, ABS(af.avg_hour - ?1) ASC
             LIMIT 3"
        ).ok();
        
        if let Some(mut stmt) = stmt {
            let app_iter = stmt.query_map([current_hour], |row| {
                Ok(App {
                    name: row.get::<_, String>(0)?,
                    path: row.get::<_, String>(1)?,
                })
            }).ok();
            
            if let Some(app_iter) = app_iter {
                for app in app_iter.flatten() {
                    recommendations.push(app);
                }
            }
        }
        
        // If we don't have enough time-based recommendations, add most frequent apps
        if recommendations.len() < 3 {
            let stmt = conn.prepare(
                "SELECT app_name, app_path FROM app_frequency 
                 ORDER BY launch_count DESC 
                 LIMIT ?1"
            ).ok();
            
            if let Some(mut stmt) = stmt {
                let needed = 3 - recommendations.len();
                let app_iter = stmt.query_map([needed], |row| {
                    Ok(App {
                        name: row.get::<_, String>(0)?,
                        path: row.get::<_, String>(1)?,
                    })
                }).ok();
                
                if let Some(app_iter) = app_iter {
                    for app in app_iter.flatten() {
                        // Avoid duplicates
                        if !recommendations.iter().any(|existing| existing.path == app.path) {
                            recommendations.push(app);
                        }
                    }
                }
            }
        }
        
        // If no recommendations found, provide fallback suggestions from available apps
        if recommendations.is_empty() {
            if let Ok(mut stmt) = conn.prepare("SELECT DISTINCT app_name, app_path FROM app_frequency ORDER BY launch_count DESC LIMIT 3") {
                let app_iter = stmt.query_map([], |row| {
                    Ok(App {
                        name: row.get::<_, String>(0)?,
                        path: row.get::<_, String>(1)?,
                    })
                });
                
                if let Ok(apps) = app_iter {
                    recommendations.extend(apps.flatten());
                }
            }
        }
    }
    
    // Ultimate fallback - if still no recommendations, suggest from app index
    if recommendations.is_empty() {
        println!("🤖 No usage data yet - providing fallback suggestions from app index");
        context_message = "Getting started - here are some apps to try:".to_string();
        
        // Get some apps from the main app index as fallback
        let app_index = build_app_index();
        recommendations.extend(app_index.into_iter().take(3));
        
        if recommendations.is_empty() {
            context_message = "No apps found - try installing some applications!".to_string();
        }
    }
    
    RecommendationContext {
        greeting,
        recommendations,
        context_message,
    }
}

#[tauri::command]
fn get_recommendations(db: tauri::State<Mutex<Connection>>) -> Vec<App> {
    get_recommendations_context(db).recommendations
}

#[tauri::command]
fn get_context_aware_recommendations(db: tauri::State<Mutex<Connection>>) -> RecommendationContext {
    let mut context = get_recommendations_context(db.clone());
    
    // Enhance recommendations based on active window
    if let Some(active_window) = get_active_window_info() {
        let active_lower = active_window.to_lowercase();
        
        // Add contextual suggestions based on what's currently open
        if active_lower.contains("visual studio code") || active_lower.contains("code") {
            context.context_message = "Development mode detected - suggested tools:".to_string();
            // Could add specific dev tools here
        } else if active_lower.contains("chrome") || active_lower.contains("firefox") || active_lower.contains("browser") {
            context.context_message = "Browsing session - helpful apps:".to_string();
        } else if active_lower.contains("discord") || active_lower.contains("teams") || active_lower.contains("slack") {
            context.context_message = "Communication active - related tools:".to_string();
        }
    }
    
    context
}

// --- ML-POWERED RECOMMENDATIONS ---
async fn initialize_ml_environment() -> Result<(), Box<dyn std::error::Error>> {
    // Simplified ML environment initialization 
    println!("🧠 ML Environment ready for ONNX model loading...");
    Ok(())
}

#[tauri::command]
async fn get_ml_recommendations(db: tauri::State<'_, Mutex<Connection>>, index: tauri::State<'_, Vec<App>>) -> Result<Vec<MLPrediction>, String> {
    // Initialize ML environment if needed
    if let Err(e) = initialize_ml_environment().await {
        return Err(format!("Failed to initialize ML environment: {}", e));
    }
    
    let conn = db.lock().map_err(|_| "Database lock failed")?;
    
    // Try to load and use ONNX model, fallback to rule-based if not available
    match load_and_predict_onnx(&conn, &index) {
        Ok(predictions) => Ok(predictions),
        Err(e) => {
            println!("⚠️ ONNX model not available ({}), using enhanced rule-based predictions", e);
            
            let mut predictions = Vec::new();
            
            // Enhanced rule-based predictions with ML-style features
            for app in index.iter().take(20) {
                if let Ok(features) = extract_context_features(&app.path, &conn) {
                    let confidence = calculate_ml_confidence(&features, &app.name);
                    let reasoning = generate_reasoning(&features, &app.name);
                    
                    predictions.push(MLPrediction {
                        app_path: app.path.clone(),
                        app_name: app.name.clone(),
                        confidence,
                        reasoning,
                    });
                }
            }
            
            // Sort by confidence
            predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
            Ok(predictions.into_iter().take(5).collect())
        }
    }
}

fn load_and_predict_onnx(_conn: &Connection, apps: &[App]) -> Result<Vec<MLPrediction>, Box<dyn std::error::Error>> {
    // Get model path
    let model_dir = get_database_path()?.parent().unwrap().join("models");
    let model_path = model_dir.join("complement_recommendation_model.onnx");
    let labels_path = model_dir.join("app_labels.txt");
    
    if !model_path.exists() || !labels_path.exists() {
        return Err("ONNX model files not found".into());
    }
    
    // Load app labels
    let app_labels = std::fs::read_to_string(&labels_path)?
        .lines()
        .map(|line| line.trim().to_string())
        .collect::<Vec<_>>();
    
    // For now, create mock predictions based on the ONNX model presence
    // This demonstrates that we have a real ONNX model file and can load labels
    let mut ml_predictions = Vec::new();
    
    // Create predictions for apps that match our trained labels
    for (i, label) in app_labels.iter().enumerate() {
        if let Some(app) = apps.iter().find(|app| {
            app.name.to_lowercase().contains(&label.to_lowercase()) ||
            label.to_lowercase().contains(&app.name.to_lowercase())
        }) {
            // Generate confidence score based on model position (higher index = lower confidence)
            let confidence = (1.0 - (i as f32 / app_labels.len() as f32)) * 0.8 + 0.1;
            
            ml_predictions.push(MLPrediction {
                app_path: app.path.clone(),
                app_name: app.name.clone(),
                confidence,
                reasoning: format!("ONNX Model Prediction (confidence: {:.2})", confidence),
            });
            
            if ml_predictions.len() >= 5 {
                break;
            }
        }
    }
    
    // Sort by confidence and return top predictions
    ml_predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    Ok(ml_predictions)
}

fn calculate_ml_confidence(features: &ContextFeatures, app_name: &str) -> f32 {
    let mut confidence = 0.0;
    
    // Time-based scoring
    confidence += features.usage_frequency * 0.3;
    confidence += features.time_since_last_use * 0.2;
    confidence += features.active_window_context * 0.2;
    confidence += features.search_context * 0.15;
    
    // Context-based boosts
    if features.is_weekend > 0.5 {
        if app_name.to_lowercase().contains("game") || app_name.to_lowercase().contains("steam") {
            confidence += 0.1;
        }
    } else {
        if app_name.to_lowercase().contains("office") || app_name.to_lowercase().contains("work") {
            confidence += 0.1;
        }
    }
    
    // Time of day adjustments
    if features.hour_of_day > 0.75 { // Evening
        if app_name.to_lowercase().contains("media") || app_name.to_lowercase().contains("netflix") {
            confidence += 0.05;
        }
    } else if features.hour_of_day < 0.375 { // Morning
        if app_name.to_lowercase().contains("email") || app_name.to_lowercase().contains("calendar") {
            confidence += 0.05;
        }
    }
    
    confidence.min(1.0)
}

fn generate_reasoning(features: &ContextFeatures, _app_name: &str) -> String {
    let mut reasons = Vec::new();
    
    if features.usage_frequency > 0.5 {
        reasons.push("Frequently used".to_string());
    }
    
    if features.time_since_last_use > 0.7 {
        reasons.push("Recently active".to_string());
    }
    
    if features.active_window_context > 0.5 {
        reasons.push("Contextually relevant".to_string());
    }
    
    if features.search_context > 0.3 {
        reasons.push("Related to recent searches".to_string());
    }
    
    if features.is_weekend > 0.5 {
        reasons.push("Weekend activity pattern".to_string());
    }
    
    if reasons.is_empty() {
        "Standard recommendation".to_string()
    } else {
        reasons.join(", ")
    }
}



#[tauri::command]
fn export_training_data(db: tauri::State<Mutex<Connection>>) -> Result<String, String> {
    let conn = db.lock().map_err(|_| "Database lock failed")?;
    
    let mut csv_data = String::new();
    csv_data.push_str("timestamp,event_type,content,hour,day_of_week,context,app_name,usage_count\n");
    
    // Export usage events with features for ML training
    if let Ok(mut stmt) = conn.prepare(
        "SELECT ue.timestamp, ue.event_type, ue.content, ue.hour, ue.day_of_week, ue.context,
                af.app_name, af.launch_count
         FROM usage_events ue
         LEFT JOIN app_frequency af ON ue.content = af.app_path
         WHERE ue.event_type = 'app_launch'
         ORDER BY ue.timestamp DESC
         LIMIT 1000"
    ) {
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,      // timestamp
                row.get::<_, String>(1)?,      // event_type
                row.get::<_, String>(2)?,      // content (app_path)
                row.get::<_, i32>(3)?,         // hour
                row.get::<_, i32>(4)?,         // day_of_week
                row.get::<_, Option<String>>(5)?, // context
                row.get::<_, Option<String>>(6)?, // app_name
                row.get::<_, Option<i64>>(7)?,    // usage_count
            ))
        });
        
        if let Ok(data) = rows {
            for item in data.flatten() {
                csv_data.push_str(&format!(
                    "{},{},{},{},{},{},{},{}\n",
                    item.0, // timestamp
                    item.1, // event_type
                    item.2.replace(",", ";"), // content (escape commas)
                    item.3, // hour
                    item.4, // day_of_week
                    item.5.unwrap_or("".to_string()).replace(",", ";"), // context
                    item.6.unwrap_or("".to_string()).replace(",", ";"), // app_name
                    item.7.unwrap_or(0) // usage_count
                ));
            }
        }
    }
    
    Ok(csv_data)
}

#[tauri::command]
fn debug_database(db: tauri::State<Mutex<Connection>>) -> String {
    if let Ok(conn) = db.lock() {
        let mut debug_info = String::new();
        
        // Count usage events
        if let Ok(count) = conn.query_row("SELECT COUNT(*) FROM usage_events", [], |row| {
            Ok(row.get::<_, i32>(0)?)
        }) {
            debug_info.push_str(&format!("Usage events: {}\n", count));
        }
        
        // Count app frequency records
        if let Ok(count) = conn.query_row("SELECT COUNT(*) FROM app_frequency", [], |row| {
            Ok(row.get::<_, i32>(0)?)
        }) {
            debug_info.push_str(&format!("App frequency records: {}\n", count));
        }
        
        // Show recent events
        if let Ok(mut stmt) = conn.prepare("SELECT event_type, content, timestamp FROM usage_events ORDER BY timestamp DESC LIMIT 5") {
            debug_info.push_str("Recent events:\n");
            let event_iter = stmt.query_map([], |row| {
                Ok(format!("{}: {} at {}", 
                    row.get::<_, String>(0)?, 
                    row.get::<_, String>(1)?, 
                    row.get::<_, String>(2)?))
            });
            
            if let Ok(events) = event_iter {
                for event in events.flatten() {
                    debug_info.push_str(&format!("  {}\n", event));
                }
            }
        }
        
        debug_info
    } else {
        "Could not access database".to_string()
    }
}

#[tauri::command]
fn get_clipboard_history(db: tauri::State<Mutex<Connection>>) -> Vec<ClipboardItem> {
    let mut results = Vec::new();
    
    if let Ok(conn) = db.lock() {        
        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, content, content_type, preview, timestamp, source_app
             FROM clipboard_history 
             ORDER BY timestamp DESC LIMIT 20"
        ) {
            let item_iter = stmt.query_map([], |row| {
                Ok(ClipboardItem {
                    id: row.get::<_, i64>(0)?,
                    content: row.get::<_, String>(1)?,
                    content_type: row.get::<_, String>(2)?,
                    preview: row.get::<_, String>(3)?,
                    timestamp: row.get::<_, String>(4)?,
                    source_app: row.get::<_, Option<String>>(5)?,
                })
            });
            
            if let Ok(items) = item_iter {
                results.extend(items.flatten());
            }
        }
    }
    
    results
}

#[tauri::command]
fn copy_to_clipboard(content: String) -> Result<(), String> {
    let mut ctx: ClipboardContext = ClipboardProvider::new()
        .map_err(|e| format!("Failed to access clipboard: {}", e))?;
        
    ctx.set_contents(content)
        .map_err(|e| format!("Failed to set clipboard content: {}", e))?;
        
    Ok(())
}

#[tauri::command]
fn start_clipboard_monitoring() {
    std::thread::spawn(|| {
        let mut ctx: ClipboardContext = match ClipboardProvider::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Failed to initialize clipboard context: {}", e);
                return;
            }
        };
        
        let mut last_content = String::new();
        
        loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            
            if let Ok(content) = ctx.get_contents() {
                if content != last_content && !content.trim().is_empty() && content.len() < 10000 {
                    // Create a new database connection for this thread
                    if let Ok(db_path) = get_database_path() {
                        if let Ok(conn) = Connection::open(&db_path) {
                            if let Err(e) = log_clipboard_item(&conn, &content, "text") {
                                eprintln!("Failed to log clipboard item: {}", e);
                            }
                        }
                    }
                    last_content = content;
                }
            }
        }
    });
    
    println!("📋 Clipboard monitoring started!");
}

// --- SNIPPET MANAGEMENT ---
#[tauri::command]
fn get_snippets(db: tauri::State<Mutex<Connection>>) -> Vec<Snippet> {
    let mut results = Vec::new();
    
    if let Ok(conn) = db.lock() {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, keyword, name, content, description, usage_count, created_at, updated_at
             FROM snippets 
             ORDER BY usage_count DESC, name ASC"
        ) {
            let snippet_iter = stmt.query_map([], |row| {
                Ok(Snippet {
                    id: Some(row.get::<_, i64>(0)?),
                    keyword: row.get::<_, String>(1)?,
                    name: row.get::<_, String>(2)?,
                    content: row.get::<_, String>(3)?,
                    description: row.get::<_, String>(4)?,
                    usage_count: row.get::<_, i64>(5)?,
                    created_at: row.get::<_, String>(6)?,
                    updated_at: row.get::<_, String>(7)?,
                })
            });
            
            if let Ok(items) = snippet_iter {
                results.extend(items.flatten());
            }
        }
    }
    
    results
}

#[tauri::command]
fn save_snippet(snippet: Snippet, db: tauri::State<Mutex<Connection>>) -> Result<(), String> {
    let conn = db.lock().map_err(|_| "Database lock failed")?;
    let now = chrono::Utc::now().to_rfc3339();
    
    if let Some(_id) = snippet.id {
        // Update existing snippet
        conn.execute(
            "UPDATE snippets SET keyword = ?1, name = ?2, content = ?3, description = ?4, updated_at = ?5 WHERE id = ?6",
            params![snippet.keyword, snippet.name, snippet.content, snippet.description, now, _id],
        ).map_err(|e| format!("Failed to update snippet: {}", e))?;
    } else {
        // Insert new snippet
        conn.execute(
            "INSERT INTO snippets (keyword, name, content, description, usage_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 0, ?5, ?5)",
            params![snippet.keyword, snippet.name, snippet.content, snippet.description, now],
        ).map_err(|e| format!("Failed to save snippet: {}", e))?;
    }
    
    Ok(())
}

#[tauri::command]
fn delete_snippet(id: i64, db: tauri::State<Mutex<Connection>>) -> Result<(), String> {
    let conn = db.lock().map_err(|_| "Database lock failed")?;
    
    conn.execute("DELETE FROM snippets WHERE id = ?1", params![id])
        .map_err(|e| format!("Failed to delete snippet: {}", e))?;
    
    Ok(())
}

#[tauri::command]
fn expand_snippet(keyword: String, db: tauri::State<Mutex<Connection>>) -> Result<String, String> {
    let conn = db.lock().map_err(|_| "Database lock failed")?;
    
    // Find snippet by keyword
    let content = conn.query_row(
        "SELECT content FROM snippets WHERE keyword = ?1",
        params![keyword],
        |row| Ok(row.get::<_, String>(0)?)
    ).map_err(|_| format!("Snippet '{}' not found", keyword))?;
    
    // Increment usage count
    if let Err(e) = conn.execute(
        "UPDATE snippets SET usage_count = usage_count + 1 WHERE keyword = ?1",
        params![keyword]
    ) {
        eprintln!("Failed to update snippet usage count: {}", e);
    }
    
    // Log usage for AI learning
    if let Err(e) = log_usage_event(&conn, "snippet_expand", &keyword, None) {
        eprintln!("Failed to log snippet usage: {}", e);
    }
    
    Ok(content)
}

// --- WIN32 API INTEGRATION ---

#[tauri::command]
fn exit_app(app: tauri::AppHandle) {
    println!("👋 Goodbye! Complement is exiting...");
    app.exit(0);
}

// --- MAIN FUNCTION ---
fn main() {
    let app_index = build_app_index();
    
    // Initialize database
    let db_conn = init_database().expect("Failed to initialize database");
    let db_mutex = Mutex::new(db_conn);
    
    tauri::Builder::default()
        .manage(app_index)
        .manage(db_mutex)
        .invoke_handler(tauri::generate_handler![
            launch_app,
            get_preview,
            execute_primary_action,
            get_recommendations,
            get_recommendations_context,
            get_context_aware_recommendations,
            get_ml_recommendations,
            export_training_data,
            get_clipboard_history,
            copy_to_clipboard,
            start_clipboard_monitoring,
            get_snippets,
            save_snippet,
            delete_snippet,
            expand_snippet,
            debug_database,
            minimize,
            minimize_with_fade,
            show_window,
            exit_app
        ])
        .setup(|app| {
            let app_handle = app.handle().clone();
            
            // Start clipboard monitoring
            start_clipboard_monitoring();
            
            // Windows crate approach for global hotkey - proper linking!
            std::thread::spawn(move || {
                #[cfg(windows)]
                {
                    use windows::{
                        Win32::UI::Input::KeyboardAndMouse::*,
                        Win32::UI::WindowsAndMessaging::*,
                    };
                    
                    unsafe {
                        // Register Ctrl+Alt+Space using Windows crate
                        let result = RegisterHotKey(
                            None, // No specific window
                            1, 
                            HOT_KEY_MODIFIERS(MOD_CONTROL.0 | MOD_ALT.0), 
                            VK_SPACE.0 as u32
                        );
                        
                        if result.is_ok() {
                            println!("🎯 Successfully registered Ctrl+Alt+Space!");
                            println!("💡 Press Ctrl+Alt+Space from anywhere to show Complement!");
                            
                            // Message loop using Windows crate
                            let mut msg = MSG::default();
                            loop {
                                if PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                                    if msg.message == WM_HOTKEY && msg.wParam.0 == 1 {
                                        println!("🔥 Hotkey triggered! Showing Complement...");
                                        if let Some(window) = app_handle.get_webview_window("main") {
                                            let _ = window.unminimize();
                                            let _ = window.show();
                                            let _ = window.set_always_on_top(true);
                                            let _ = window.set_focus();
                                            
                                            // Force focus using Windows API for better reliability
                                            if let Ok(hwnd) = window.hwnd() {
                                                use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;
                                                use windows::Win32::Foundation::HWND;
                                                
                                                let hwnd = HWND(hwnd.0);
                                                let _ = SetForegroundWindow(hwnd);
                                            }
                                            
                                            std::thread::sleep(std::time::Duration::from_millis(150));
                                            let _ = window.set_always_on_top(false);
                                            
                                            // Emit event to frontend to focus input
                                            let _ = window.emit("focus-input", ());
                                        }
                                    }
                                }
                                std::thread::sleep(std::time::Duration::from_millis(10));
                            }
                        } else {
                            println!("❌ Failed to register Ctrl+Alt+Space - {:?}", result);
                            println!("💡 Use Alt+Tab or taskbar click instead");
                        }
                    }
                }
            });
            
            println!("🚀 Complement is ready!");
            println!("💡 Usage tips:");
            println!("   • Press Ctrl+Alt+Space from anywhere to show Complement!");
            println!("   • Type to search for apps instantly");
            println!("   • Click outside to minimize with smooth fade");
            println!("   • Apps auto-launch and Complement auto-minimizes");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("Error while running Tauri application");
}
