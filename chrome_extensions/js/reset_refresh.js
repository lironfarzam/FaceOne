/**
 * Reset Refresh Loop Helper Script
 * 
 * This script can be loaded in Chrome DevTools console to reset the refresh loop detection
 * in the FaceOne extension.
 * 
 * To use:
 * 1. Open Chrome DevTools on a Facebook page where FaceOne is active (press F12)
 * 2. Copy and paste this entire script into the console
 * 3. Press Enter to run it
 */

(function() {
    // Clear all storage keys related to refresh loop detection
    try {
        // Clear session storage
        sessionStorage.removeItem('faceone_page_refresh');
        
        // Clear local storage keys
        localStorage.removeItem('faceone_disabled_until');
        localStorage.removeItem('faceone_permanently_disabled');
        localStorage.removeItem('faceone_permanent_disable');
        
        console.log('%c[FaceOne] Refresh loop detection reset successfully', 'color: green; font-weight: bold');
        console.log('%c[FaceOne] Please reload the page for changes to take effect', 'color: blue');
        
        // Optional: Force reload the page
        if (confirm('Reload the page to apply changes?')) {
            location.reload();
        }
    } catch (error) {
        console.error('%c[FaceOne] Error resetting refresh loop detection:', 'color: red', error);
    }
})(); 