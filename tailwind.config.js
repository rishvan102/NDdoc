/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./**/*.js"],
  theme: {
    extend: {
      colors: { brand: { DEFAULT: "#ffb703", dark: "#fb8500" } }
    }
  },
  plugins: []
};
