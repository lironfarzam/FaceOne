document.addEventListener('DOMContentLoaded', () => {
  console.log("🔄 DOMContentLoaded event fired");
  // Check if TensorFlow is ready
  if (window.tfReady) {
    console.log("✅ TensorFlow is ready");
    showControls();
  } else {
    console.log("⏳ Waiting for TensorFlow to be ready");
    document.getElementById('loading').textContent = 'Loading TensorFlow...';
    window.addEventListener('tfReady', () => {
      console.log("✅ TensorFlow is ready (event)");
      showControls();
    });
  }
});

function showControls() {
  console.log("🔧 Showing controls");
  document.getElementById('loading').style.display = 'none';
  document.getElementById('controls').style.display = 'block';
}

document.getElementById('start').addEventListener('click', () => {
  console.log("▶️ Start button clicked");
  chrome.runtime.sendMessage({ action: 'startFaceReplacement' });
  document.getElementById('status-text').textContent = 'Running';
});

document.getElementById('stop').addEventListener('click', () => {
  console.log("⏹️ Stop button clicked");
  chrome.runtime.sendMessage({ action: 'stopFaceReplacement' });
  document.getElementById('status-text').textContent = 'Stopped';
});
