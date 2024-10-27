const languageMap = {
  'english': 'en',
  'french': 'fr',
  'hindi': 'hi',
  'spanish': 'es',
  'german': 'de',
  'italian': 'it',
  'japanese': 'ja',
  'korean': 'ko',
  'chinese': 'zh',
  'russian': 'ru',
  // Indian languages
  'punjabi': 'pa',
  'telugu': 'te',
  'tamil': 'ta',
  'marathi': 'mr',
  'gujarati': 'gu',
  'kannada': 'kn',
  'malayalam': 'ml',
  'bengali': 'bn',
  'odia': 'or',
  'urdu': 'ur'
  // You can add more languages as needed
};

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "translateText",
    title: "Translate text",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "translateText") {
    const text = info.selectionText;
    chrome.storage.sync.get('targetLang', function(data) {
      let targetLang = data.targetLang || 'en'; // Default to English if no language is set
      targetLang = targetLang.toLowerCase();
      // If stored value is a full language name, convert to code
      if (languageMap[targetLang]) {
        targetLang = languageMap[targetLang];
      }
      translateText(text, targetLang, tab.id);
    });
  }
});

function translateText(text, targetLang, tabId) {
  const apiUrl = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${targetLang}&dt=t&q=${encodeURIComponent(text)}`;

  fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
      const translation = data[0][0][0];
      chrome.tabs.sendMessage(tabId, { action: "showTranslation", translation: translation });
    })
    .catch(error => console.error("Translation error:", error));
}
