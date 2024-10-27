let lastSelectedText = '';
let translationTimeout;

document.addEventListener('selectionchange', () => {
  clearTimeout(translationTimeout);
  translationTimeout = setTimeout(() => {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText && selectedText !== lastSelectedText) {
      lastSelectedText = selectedText;
      chrome.runtime.sendMessage({ action: "translateToHindi", text: selectedText });
    }
  }, 500); // Wait for 500ms after selection stops changing
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "showTranslation") {
    showTranslationPopup(request.translation);
  }
});

function showTranslationPopup(translation) {
  const popup = document.createElement("div");
  popup.style.cssText = `
    position: fixed;
    top: 10px;
    right: 10px;
    background-color: white;
    border: 1px solid black;
    padding: 10px;
    z-index: 9999;
    font-family: Arial, sans-serif;
  `;
  popup.textContent = translation;

  const closeButton = document.createElement("button");
  closeButton.textContent = "Close";
  closeButton.style.marginTop = "10px";
  closeButton.onclick = () => document.body.removeChild(popup);

  popup.appendChild(document.createElement("br"));
  popup.appendChild(closeButton);

  document.body.appendChild(popup);
}
