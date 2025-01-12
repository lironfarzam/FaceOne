console.log("🔄 TF Loader initializing...");

function injectScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.async = true;
    script.src = chrome.runtime.getURL(src);
    script.onload = () => {
      console.log(`📦 Script loaded: ${src}`);
      resolve();
    };
    script.onerror = reject;
    (document.head || document.documentElement).appendChild(script);
  });
}

async function initTensorFlow() {
  try {
    // Load scripts sequentially
    await injectScript('libs/tf.js');
    await injectScript('libs/tf-backend-wasm.js');
    
    // Wait for TF to be available in global scope
    await new Promise(resolve => {
      const checkTf = () => {
        if (window.tf) {
          resolve();
        } else {
          setTimeout(checkTf, 100);
        }
      };
      checkTf();
    });

    // Initialize WASM backend
    await tf.setBackend('wasm');
    await tf.ready();
    
    console.log("✅ WASM backend initialized");
    console.log("🔧 Current backend:", tf.getBackend());
    
    // Signal that TF is ready
    window.tfReady = true;
    window.dispatchEvent(new Event('tfReady'));
    
  } catch (err) {
    console.error("❌ Error initializing TF:", err);
  }
}

// Check if document is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initTensorFlow);
} else {
  initTensorFlow();
}
