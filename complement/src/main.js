const { invoke } = window.__TAURI__.core;

let commandInputEl;

window.addEventListener("DOMContentLoaded", () => {
  commandInputEl = document.querySelector("#command-input");

  // Listen for the Enter key in the input field
  commandInputEl.addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      let response = await invoke("handle_command", { input: commandInputEl.value })
      console.log(response);
      commandInputEl.value = "";
    }
  })
});
