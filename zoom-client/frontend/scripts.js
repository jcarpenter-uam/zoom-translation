document.addEventListener("DOMContentLoaded", async () => {
  const outputEl = document.getElementById("linesTranscript");

  async function configureZoomSdk() {
    try {
      const configResponse = await zoomSdk.config({
        version: "0.16",
        popoutSize: { width: 480, height: 360 },
        capabilities: ["shareApp"],
      });
      console.log("Zoom SDK Configuration successful", configResponse);
      return configResponse;
    } catch (error) {
      console.error("Error configuring the Zoom SDK:", error);
      throw error;
    }
  }

  async function fetchConfig() {
    try {
      const response = await fetch("/api/config");
      if (!response.ok) {
        throw new Error(
          `Configuration request failed with status ${response.status}`,
        );
      }
      return await response.json();
    } catch (error) {
      console.error("Failed to fetch server configuration:", error);
      document.getElementById("transcription-indicator").className =
        "status-indicator disconnected";
      document.getElementById("translation-indicator").className =
        "status-indicator disconnected";
      throw error;
    }
  }

  async function initializeApp() {
    const [zoomResult, configResult] = await Promise.allSettled([
      configureZoomSdk(),
      fetchConfig(),
    ]);

    if (zoomResult.status === "rejected") {
      console.warn("Zoom SDK failed to initialize, but the app will proceed.");
    }

    if (configResult.status === "fulfilled") {
      const config = configResult.value;

      connectWebSocket({
        name: "Transcription",
        url: config.transcription_url,
        indicatorEl: document.getElementById("transcription-indicator"),
      });

      connectWebSocket({
        name: "Translation",
        url: config.translation_url,
        indicatorEl: document.getElementById("translation-indicator"),
      });
    } else {
      console.error(
        "Critical error: Could not fetch server configuration. WebSockets will not connect.",
      );
    }
  }

  function updateUI(message) {
    if (!message.id) return;

    const isTranscription = message.id.startsWith("t-");
    const sentenceNum = message.id.split("-")[1];
    const pairId = `pair-${sentenceNum}`;
    const targetId = message.id;

    let pairContainer = document.getElementById(pairId);

    if (!pairContainer) {
      pairContainer = document.createElement("div");
      pairContainer.className = "sentence-pair";
      pairContainer.id = pairId;

      pairContainer.innerHTML = `
        <div class="speaker-name"></div>
        <div class="text-group">
          <p class="translation" id="tr-${sentenceNum}">
            <span class="spinner"></span>
            <span class="text-content"></span>
            <span class="timestamp-display"></span>
          </p>
          <p class="transcription" id="t-${sentenceNum}">
            <span class="text-content"></span>
          </p>
        </div>
      `;
      outputEl.appendChild(pairContainer);
    }

    const speakerNameEl = pairContainer.querySelector(".speaker-name");
    if (speakerNameEl && message.userName && !speakerNameEl.textContent) {
      speakerNameEl.textContent = `${message.userName}:`;
    }

    const targetEl = document.getElementById(targetId);
    const textContentEl = targetEl.querySelector(".text-content");
    if (textContentEl) {
      textContentEl.textContent = message.data;
    }

    if (message.type === "final") {
      targetEl.classList.add("finalized");
      if (!isTranscription) {
        const spinner = targetEl.querySelector(".spinner");
        if (spinner) spinner.remove();
        if (message.timestamp) {
          const timestampNode = targetEl.querySelector(".timestamp-display");
          if (timestampNode) {
            const d = new Date(message.timestamp);
            timestampNode.textContent = `[${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}]`;
          }
        }
      }
    }
    outputEl.parentElement.scrollTop = outputEl.parentElement.scrollHeight;
  }

  function connectWebSocket(config) {
    const ws = new WebSocket(config.url);
    ws.onopen = () => {
      config.indicatorEl.className = "status-indicator connected";
    };
    ws.onerror = (error) =>
      console.error(`WebSocket Error for ${config.name}:`, error);
    ws.onclose = () => {
      config.indicatorEl.className = "status-indicator disconnected";
      setTimeout(() => {
        config.indicatorEl.className = "status-indicator connecting";
        connectWebSocket(config);
      }, 5000);
    };
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      updateUI(message);
    };
  }

  initializeApp();
});
