// Remembers form values between page loads (browser localStorage).
// Storage can be blocked (private window), so every call is wrapped and falls back quietly.
const PREFIX = 'dpv.';

export const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(PREFIX + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (e) {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(PREFIX + key, JSON.stringify(value));
    } catch (e) {
      // blocked storage: the page just will not remember anything
    }
  },
};
