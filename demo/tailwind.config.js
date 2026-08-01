/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0d1117',
        surface: '#161b22',
        border: '#30363d',
        text: '#e6edf3',
        accentBlue: '#58a6ff',
        accentGreen: '#3fb950',
        accentRed: '#f85149',
        accentYellow: '#d29922',
        accentPurple: '#bc8cff',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
