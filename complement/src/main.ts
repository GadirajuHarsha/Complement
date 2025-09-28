import { invoke } from "@tauri-apps/api/core";
import { Window, LogicalSize, LogicalPosition } from "@tauri-apps/api/window";
import { listen } from "@tauri-apps/api/event";

const appWindow = Window.getCurrent();

// Make the window click-through by default
appWindow.setIgnoreCursorEvents(true);

interface App {
  name: string;
  path: string;
}

interface RecommendationContext {
  greeting: string;
  recommendations: App[];
  context_message: string;
}

interface ClipboardItem {
  id: number;
  content: string;
  content_type: string;
  preview: string;
  timestamp: string;
  source_app?: string;
}

interface Snippet {
  id?: number;
  keyword: string;
  name: string;
  content: string;
  description: string;
  usage_count: number;
  created_at: string;
  updated_at: string;
}

interface MLPrediction {
  app_path: string;
  app_name: string;
  confidence: number;
  reasoning: string;
}

type CommandResult =
  | { type: 'Apps', payload: App[] }
  | { type: 'Text', payload: string }
  | { type: 'Recommendations', payload: App[], greeting?: string, contextMessage?: string }
  | { type: 'Clipboard', payload: ClipboardItem[] }
  | { type: 'Snippets', payload: Snippet[] }
  | { type: 'MLRecommendations', payload: MLPrediction[] };

let commandInputEl: HTMLInputElement | null;
let resultsContainerEl: HTMLElement | null;
let containerEl: HTMLElement | null;
let selectedIndex = 0;
let currentResults: App[] = [];

// Reusable fade-out animation for minimize
async function minimizeWithFade() {
  const commandContainer = document.getElementById("command-container");
  if (commandContainer) {
    commandContainer.style.transition = "opacity 0.25s";
    commandContainer.style.opacity = "0";
    setTimeout(() => {
      invoke("minimize");
      clearCommandBar(); // Clear after minimize
      commandContainer.style.opacity = "1";
      commandContainer.style.transition = "";
    }, 250);
  } else {
    // Fallback if no animation container
    await invoke("minimize");
    clearCommandBar(); // Clear after minimize
  }
}

// Special exit animation with scale and fade
async function exitWithAnimation() {
  const commandContainer = document.getElementById("command-container");
  if (commandContainer) {
    // Create a dramatic exit animation
    commandContainer.style.transition = "all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)";
    commandContainer.style.transform = "translate(-50%, -50%) scale(0.8)";
    commandContainer.style.opacity = "0";
    commandContainer.style.filter = "blur(2px)";

    setTimeout(() => {
      invoke("exit_app");
    }, 500);
  } else {
    // Fallback if no animation container
    await invoke("exit_app");
  }
}

function updateWindowSize() {
  if (!containerEl || !resultsContainerEl || !commandInputEl) return;

  const contentHeight = commandInputEl.offsetHeight + resultsContainerEl.offsetHeight;

  const newHeight = contentHeight > 50 ? contentHeight + 20 : 80;
  const newWidth = 715;

  appWindow.setSize(new LogicalSize(newWidth, newHeight));
  appWindow.setPosition(new LogicalPosition(window.innerWidth / 2 - newWidth / 2, window.innerHeight / 2 - newHeight / 2));
}


async function updatePreview() {
  if (!commandInputEl || !resultsContainerEl) return;

  let result: CommandResult;
  const query = commandInputEl.value.trim();

  // If search is empty, get AI recommendations
  if (query === "") {
    const context = await invoke<RecommendationContext>("get_context_aware_recommendations");
    console.log("🤖 Received recommendations:", context);
    result = {
      type: 'Recommendations',
      payload: context.recommendations,
      greeting: context.greeting,
      contextMessage: context.context_message
    };
  }
  // Check for clipboard history commands
  else if (query === "c" || query === "clip" || query === "clipboard" || query.startsWith("c ")) {
    const clipboardItems = await invoke<ClipboardItem[]>("get_clipboard_history");
    result = {
      type: 'Clipboard',
      payload: clipboardItems
    };
  }
  // Check for snippet commands
  else if (query === "s" || query === "snippet" || query === "snippets" || query.startsWith("s ")) {
    const snippets = await invoke<Snippet[]>("get_snippets");
    result = {
      type: 'Snippets',
      payload: snippets
    };
  }
  // Check for ML commands
  else if (query === "ml" || query === "ai" || query === "predict" || query === "smart") {
    const mlPredictions = await invoke<MLPrediction[]>("get_ml_recommendations");
    result = {
      type: 'MLRecommendations',
      payload: mlPredictions
    };
  }
  // Default to normal search
  else {
    result = await invoke<CommandResult>("get_preview", { query: commandInputEl.value });
  }

  resultsContainerEl.innerHTML = "";

  // Check that we got the 'Apps' variant and then use its payload
  const commandContainer = document.getElementById("command-container");
  let resultsCount = 0;
  // Align results directly below the command bar inside the container
  resultsContainerEl.style.position = "static";
  resultsContainerEl.style.width = "100%";
  resultsContainerEl.style.marginTop = "8px";
  resultsContainerEl.style.left = "unset";
  resultsContainerEl.style.top = "unset";
  resultsContainerEl.style.transform = "unset";

  // Reset selection when results change
  selectedIndex = 0;

  if (result.type === 'Apps') {
    currentResults = result.payload;
    currentDisplayType = 'apps';
    currentClipboardItems = [];
    resultsCount = result.payload.length;
    if (resultsCount > 0) {
      commandContainer?.classList.add("has-results");
    } else {
      commandContainer?.classList.remove("has-results");
    }
    result.payload.forEach((app, index) => {
      const resultEl = document.createElement("div");
      resultEl.classList.add("result-item");
      if (index === selectedIndex) {
        resultEl.classList.add("selected");
      }
      resultEl.textContent = app.name;
      resultEl.addEventListener('click', () => {
        invoke('launch_app', { path: app.path });
        clearCommandBar(); // Clear after clicking to launch app
      });
      resultsContainerEl?.appendChild(resultEl);
    });
  } else if (result.type === 'Text') {
    currentResults = [];
    currentDisplayType = 'other';
    currentClipboardItems = [];
    commandContainer?.classList.remove("has-results");
    const resultEl = document.createElement("div");
    resultEl.classList.add("result-item", "selected");
    resultEl.textContent = result.payload;
    resultsContainerEl.appendChild(resultEl);
  } else if (result.type === 'Recommendations') {
    currentResults = result.payload;
    currentDisplayType = 'apps';
    currentClipboardItems = [];
    resultsCount = result.payload.length;
    if (resultsCount > 0) {
      commandContainer?.classList.add("has-results");

      // Add greeting and context headers for recommendations
      const greetingEl = document.createElement("div");
      greetingEl.classList.add("recommendations-greeting");
      greetingEl.textContent = result.greeting || "Welcome back!";
      resultsContainerEl.appendChild(greetingEl);

      const headerEl = document.createElement("div");
      headerEl.classList.add("recommendations-header");
      headerEl.textContent = result.contextMessage || "Suggested for you:";
      resultsContainerEl.appendChild(headerEl);

      result.payload.forEach((app, index) => {
        const resultEl = document.createElement("div");
        resultEl.classList.add("result-item", "recommendation");
        if (index === selectedIndex) {
          resultEl.classList.add("selected");
        }
        resultEl.textContent = app.name;
        resultEl.addEventListener('click', () => {
          invoke('launch_app', { path: app.path });
          clearCommandBar(); // Clear after clicking to launch app
        });
        resultsContainerEl?.appendChild(resultEl);
      });
    } else {
      commandContainer?.classList.remove("has-results");
    }
  } else if (result.type === 'Clipboard') {
    // Store clipboard items separately from app results
    currentResults = [];
    currentDisplayType = 'clipboard';
    currentClipboardItems = result.payload;
    resultsCount = result.payload.length;

    if (resultsCount > 0) {
      commandContainer?.classList.add("has-results");

      // Add clipboard header
      const headerEl = document.createElement("div");
      headerEl.classList.add("clipboard-header");
      headerEl.textContent = "📋 Clipboard History";
      resultsContainerEl.appendChild(headerEl);

      result.payload.forEach((item, index) => {
        const resultEl = document.createElement("div");
        resultEl.classList.add("result-item", "clipboard-item");
        if (index === selectedIndex) {
          resultEl.classList.add("selected");
        }

        // Create clipboard item layout
        const previewEl = document.createElement("div");
        previewEl.classList.add("clipboard-preview");
        previewEl.textContent = item.preview;

        const metaEl = document.createElement("div");
        metaEl.classList.add("clipboard-meta");
        const timeAgo = new Date(item.timestamp).toLocaleTimeString();
        metaEl.textContent = `${item.content_type} • ${timeAgo}${item.source_app ? ` • ${item.source_app}` : ''}`;

        resultEl.appendChild(previewEl);
        resultEl.appendChild(metaEl);

        resultEl.addEventListener('click', () => {
          invoke('copy_to_clipboard', { content: item.content });
          clearCommandBar(); // Clear after copying
        });
        resultsContainerEl?.appendChild(resultEl);
      });
    } else {
      commandContainer?.classList.remove("has-results");
      const emptyEl = document.createElement("div");
      emptyEl.classList.add("result-item");
      emptyEl.textContent = "No clipboard history available";
      resultsContainerEl.appendChild(emptyEl);
    }
  } else if (result.type === 'Snippets') {
    // Store snippets separately from app results
    currentResults = [];
    currentDisplayType = 'snippets';
    currentSnippetItems = result.payload;
    resultsCount = result.payload.length;

    if (resultsCount > 0) {
      commandContainer?.classList.add("has-results");

      // Add snippets header
      const headerEl = document.createElement("div");
      headerEl.classList.add("snippets-header");
      headerEl.textContent = "📝 Available Snippets";
      resultsContainerEl.appendChild(headerEl);

      result.payload.forEach((snippet, index) => {
        const resultEl = document.createElement("div");
        resultEl.classList.add("result-item", "snippet-item");
        if (index === selectedIndex) {
          resultEl.classList.add("selected");
        }

        // Create snippet item layout
        const nameEl = document.createElement("div");
        nameEl.classList.add("snippet-name");
        nameEl.textContent = `${snippet.name} (:${snippet.keyword})`;

        const previewEl = document.createElement("div");
        previewEl.classList.add("snippet-preview");
        const preview = snippet.content.length > 50 ?
          snippet.content.substring(0, 47) + "..." :
          snippet.content;
        previewEl.textContent = preview;

        const metaEl = document.createElement("div");
        metaEl.classList.add("snippet-meta");
        metaEl.textContent = `Used ${snippet.usage_count} times${snippet.description ? ` • ${snippet.description}` : ''}`;

        resultEl.appendChild(nameEl);
        resultEl.appendChild(previewEl);
        resultEl.appendChild(metaEl);

        resultEl.addEventListener('click', () => {
          invoke('expand_snippet', { keyword: snippet.keyword });
          clearCommandBar(); // Clear after expanding
        });
        resultsContainerEl?.appendChild(resultEl);
      });
    } else {
      commandContainer?.classList.remove("has-results");
      const emptyEl = document.createElement("div");
      emptyEl.classList.add("result-item");
      emptyEl.textContent = "No snippets available - create one with :keyword";
      resultsContainerEl.appendChild(emptyEl);
    }
  } else if (result.type === 'MLRecommendations') {
    // Store ML predictions
    currentResults = [];
    currentDisplayType = 'ml';
    currentMLItems = result.payload;
    resultsCount = result.payload.length;

    if (resultsCount > 0) {
      commandContainer?.classList.add("has-results");

      // Add ML header
      const headerEl = document.createElement("div");
      headerEl.classList.add("ml-header");
      headerEl.textContent = "🧠 ONNX Neural Network Predictions";
      resultsContainerEl.appendChild(headerEl);

      result.payload.forEach((prediction, index) => {
        const resultEl = document.createElement("div");
        resultEl.classList.add("result-item", "ml-item");
        if (index === selectedIndex) {
          resultEl.classList.add("selected");
        }

        // Create ML prediction layout
        const nameEl = document.createElement("div");
        nameEl.classList.add("ml-app-name");
        nameEl.textContent = prediction.app_name;

        const confidenceEl = document.createElement("div");
        confidenceEl.classList.add("ml-confidence");
        const confidencePercent = Math.round(prediction.confidence * 100);
        confidenceEl.textContent = `${confidencePercent}% confidence`;

        const reasoningEl = document.createElement("div");
        reasoningEl.classList.add("ml-reasoning");
        reasoningEl.textContent = prediction.reasoning;

        resultEl.appendChild(nameEl);
        resultEl.appendChild(confidenceEl);
        resultEl.appendChild(reasoningEl);

        resultEl.addEventListener('click', () => {
          invoke('launch_app', { path: prediction.app_path });
          clearCommandBar();
        });
        resultsContainerEl?.appendChild(resultEl);
      });
    } else {
      commandContainer?.classList.remove("has-results");
      const emptyEl = document.createElement("div");
      emptyEl.classList.add("result-item");
      emptyEl.textContent = "🤖 No ML predictions available - ONNX model may be loading";
      resultsContainerEl.appendChild(emptyEl);
    }
  }
  // Dynamically set --results-height for smooth border growth
  // Add extra height for headers if present
  const hasRecommendationHeaders = result.type === 'Recommendations' && resultsCount > 0;
  const hasClipboardHeader = result.type === 'Clipboard' && resultsCount > 0;
  const hasSnippetsHeader = result.type === 'Snippets' && resultsCount > 0;
  let headerHeight = 0;
  if (hasRecommendationHeaders) {
    headerHeight = 60; // greeting + context header
  } else if (hasClipboardHeader || hasSnippetsHeader) {
    headerHeight = 40; // clipboard or snippets header
  }

  // Base item height varies by type
  let itemHeight = 45; // Default for apps
  if (result.type === 'Clipboard') {
    itemHeight = 60; // Clipboard items are taller
  } else if (result.type === 'Snippets') {
    itemHeight = 75; // Snippet items are tallest (name + preview + meta)
  }
  const resultsHeight = (resultsCount * itemHeight) + headerHeight;
  commandContainer?.style.setProperty('--results-height', `${resultsHeight}px`);
  updateWindowSize();
}

function updateSelection() {
  if (!resultsContainerEl) return;

  const resultItems = resultsContainerEl.querySelectorAll('.result-item');
  resultItems.forEach((item, index) => {
    item.classList.toggle('selected', index === selectedIndex);
  });
}

function moveSelectionUp() {
  let totalItems = 0;
  if (currentDisplayType === 'clipboard') {
    totalItems = currentClipboardItems.length;
  } else if (currentDisplayType === 'snippets') {
    totalItems = currentSnippetItems.length;
  } else if (currentDisplayType === 'ml') {
    totalItems = currentMLItems.length;
  } else {
    totalItems = currentResults.length;
  }

  if (totalItems === 0) return;
  selectedIndex = selectedIndex > 0 ? selectedIndex - 1 : totalItems - 1;
  updateSelection();
}

function moveSelectionDown() {
  let totalItems = 0;
  if (currentDisplayType === 'clipboard') {
    totalItems = currentClipboardItems.length;
  } else if (currentDisplayType === 'snippets') {
    totalItems = currentSnippetItems.length;
  } else if (currentDisplayType === 'ml') {
    totalItems = currentMLItems.length;
  } else {
    totalItems = currentResults.length;
  }

  if (totalItems === 0) return;
  selectedIndex = selectedIndex < totalItems - 1 ? selectedIndex + 1 : 0;
  updateSelection();
}

// Clear command bar and reset state
function clearCommandBar() {
  if (!commandInputEl || !resultsContainerEl) return;

  commandInputEl.value = "";
  resultsContainerEl.innerHTML = "";
  currentResults = [];
  currentClipboardItems = [];
  currentSnippetItems = [];
  currentMLItems = [];
  currentDisplayType = 'other';
  selectedIndex = 0;

  const commandContainer = document.getElementById("command-container");
  commandContainer?.classList.remove("has-results");
  commandContainer?.style.setProperty('--results-height', '0px');

  updateWindowSize();
}

// Store current display type for Enter key handling
let currentDisplayType: 'apps' | 'clipboard' | 'snippets' | 'ml' | 'other' = 'other';
let currentClipboardItems: ClipboardItem[] = [];
let currentSnippetItems: Snippet[] = [];
let currentMLItems: MLPrediction[] = [];

// This new function handles the Enter key
async function handleEnterKey() {
  if (!commandInputEl) return;
  const value = commandInputEl.value.trim().toLowerCase();
  if (value === "min" || value === "minimize") {
    minimizeWithFade();
    return;
  }

  if (value === "exit") {
    exitWithAnimation();
    return;
  }

  // Handle clipboard selection
  if (currentDisplayType === 'clipboard' && currentClipboardItems.length > 0 && selectedIndex < currentClipboardItems.length) {
    const selectedItem = currentClipboardItems[selectedIndex];
    await invoke('copy_to_clipboard', { content: selectedItem.content });
    clearCommandBar(); // Clear after copying
  }
  // Handle snippet selection
  else if (currentDisplayType === 'snippets' && currentSnippetItems.length > 0 && selectedIndex < currentSnippetItems.length) {
    const selectedSnippet = currentSnippetItems[selectedIndex];
    await invoke('expand_snippet', { keyword: selectedSnippet.keyword });
    clearCommandBar(); // Clear after expanding
  }
  // Handle ML selection
  else if (currentDisplayType === 'ml' && currentMLItems.length > 0 && selectedIndex < currentMLItems.length) {
    const selectedMLItem = currentMLItems[selectedIndex];
    await invoke('launch_app', { path: selectedMLItem.app_path });
    clearCommandBar(); // Clear after launching app
  }
  // Handle app selection  
  else if (currentDisplayType === 'apps' && currentResults.length > 0 && selectedIndex < currentResults.length) {
    const selectedApp = currentResults[selectedIndex];
    await invoke('launch_app', { path: selectedApp.path });
    clearCommandBar(); // Clear after launching app
  } else {
    // Fallback to the original behavior for other commands
    await invoke("execute_primary_action", { command: commandInputEl.value });
    clearCommandBar(); // Clear after executing command
  }
}

window.addEventListener("DOMContentLoaded", () => {
  commandInputEl = document.querySelector("#command-input");
  resultsContainerEl = document.querySelector("#results-container");
  containerEl = document.querySelector(".container");

  // Listen for backend-triggered fade minimize events
  listen("minimize-with-fade", () => {
    minimizeWithFade();
  });

  // Listen for focus-input event from hotkey trigger
  listen("focus-input", () => {
    commandInputEl?.focus();
  });

  // Global shortcut (Ctrl+`) is registered on the Rust backend

  // Overlay click to minimize with fade out
  const overlay = document.getElementById("overlay");
  const commandContainer = document.getElementById("command-container");
  overlay?.addEventListener("mousedown", (e) => {
    if (!commandContainer) return;
    // Only minimize if click is outside the commandContainer
    const rect = commandContainer.getBoundingClientRect();
    const x = e.clientX, y = e.clientY;
    if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
      minimizeWithFade();
    }
  });

  // When input is focused, make window interactive
  commandInputEl?.addEventListener("focus", () => {
    appWindow.setIgnoreCursorEvents(false);
  });
  // When input loses focus, make window click-through again
  commandInputEl?.addEventListener("blur", () => {
    appWindow.setIgnoreCursorEvents(true);
  });

  // 'input' event is for live previews
  commandInputEl?.addEventListener("input", updatePreview);

  // 'keydown' event is for executing the final command and navigation
  commandInputEl?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      handleEnterKey();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      moveSelectionUp();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      moveSelectionDown();
    }
  });

  // Load initial recommendations on startup
  updatePreview();
  updateWindowSize();
});