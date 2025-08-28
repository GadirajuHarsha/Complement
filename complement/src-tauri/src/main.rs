// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use meval::eval_str;

// This is our new, main command handler.
#[tauri::command]
fn handle_command(input: String) -> String {
    // try to evaluate the input string as a math expression.
    match eval_str(&input) {
        // if it's successful, we get a result.
        Ok(result) => {
            // check if result is an integer (no fractional part)
            if result.fract() == 0.0 {
                // if yes, format it as a whole number.
                format!("= {}", result as i64)
            } else {
                // format it as a decimal.
                format!("= {}", result)
            }
        }
        // if it fails, it's not a math expression.
        Err(_) => input,
    }
}

fn main() {
    tauri::Builder::default()
        // Register our new command so the frontend can call it.
        .invoke_handler(tauri::generate_handler![handle_command])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
