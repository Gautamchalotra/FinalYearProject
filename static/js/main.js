/** 
 * Custom Scripts for MediOptima
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log("MediOptima Frontend Loaded Successfully.");
    
    const themeToggleBtn = document.getElementById('theme-toggle');
    if(themeToggleBtn) {
        // Check for saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
            themeToggleBtn.innerHTML = ' Light Mode';
        }
        
        themeToggleBtn.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            
            // Save preference
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            
            // Update button text
            themeToggleBtn.innerHTML = isDark ? ' Light Mode' : ' Dark Mode';
        });
    }
});
