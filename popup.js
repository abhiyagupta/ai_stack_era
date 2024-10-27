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

document.addEventListener('DOMContentLoaded', function() {
  const targetLangInput = document.getElementById('targetLang');
  const saveButton = document.getElementById('saveButton');

  // Load saved language
  chrome.storage.sync.get('targetLang', function(data) {
    if (data.targetLang) {
      targetLangInput.value = data.targetLang;
    }
  });

  saveButton.addEventListener('click', function() {
    let targetLang = targetLangInput.value.trim().toLowerCase();
    if (targetLang) {
      // If input is a full language name, convert to code
      if (languageMap[targetLang]) {
        targetLang = languageMap[targetLang];
      }
      chrome.storage.sync.set({targetLang: targetLang}, function() {
        alert('Target language saved: ' + targetLang);
      });
    } else {
      alert('Please enter a valid language name or code.');
    }
  });
});
