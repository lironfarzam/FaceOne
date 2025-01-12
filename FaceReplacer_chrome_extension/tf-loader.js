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
    console.log("📥 Loading TensorFlow.js...");
    await injectScript('libs/tf.js');
    console.log("✅ TensorFlow.js loaded");

    console.log("📥 Loading TensorFlow WASM backend...");
    await injectScript('libs/tf-backend-wasm.js');
    console.log("✅ TensorFlow WASM backend loaded");
    
    // Wait for TF to be available in global scope
    console.log("⏳ Waiting for TensorFlow to be available in global scope...");
    await new Promise(resolve => {
      const checkTf = () => {
        if (window.tf) {
          console.log("🔧 TensorFlow is now available in global scope");
          resolve();
        } else {
          console.log("⏳ TensorFlow not yet available, retrying...");
          setTimeout(checkTf, 100);
        }
      };
      checkTf();
    });

    console.log("🔧 TensorFlow loaded in global scope");

    // Initialize WASM backend
    console.log("🔧 Initializing WASM backend...");
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
