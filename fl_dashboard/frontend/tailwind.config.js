/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                dark: {
                    bg: '#1e2129',
                    card: '#252932',
                    sidebar: '#191c24',
                    border: '#2d323e'
                }
            }
        },
    },
    plugins: [],
}
