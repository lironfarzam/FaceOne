// Background script for the extension

chrome.runtime.onInstalled.addListener(() => {
  // Register content script
  chrome.scripting.registerContentScripts([{
    id: 'face-replacer',
    matches: ['<all_urls>'],
    js: ['tf-loader.js', 'content-script.js'],
    runAt: 'document_start'
  }]);
});

// Handle script injection requests from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'INJECT_SCRIPT') {
    chrome.scripting.executeScript({
      target: { tabId: sender.tab.id },
      files: [request.script]
    });
    sendResponse({success: true});
    return true;
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "loadFaceAPI") {
    chrome.scripting.executeScript(
      {
        target: { tabId: sender.tab.id },
        files: ["libs/face-api.min.js"], // Path to your face-api.min.js
      },
      () => {
        console.log("face-api.min.js injected");
        sendResponse();
      }
    );
    return true; // Indicates the response will be sent asynchronously
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'startFaceReplacement') {
    // Logic to start face replacement
    console.log('Face replacement started');
  } else if (message.action === 'stopFaceReplacement') {
    // Logic to stop face replacement
    console.log('Face replacement stopped');
  }
  sendResponse();
});
