import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// jsdom does not implement scrollIntoView — provide a no-op so citation
// navigation does not throw during tests.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}

// Ensure the DOM is cleaned between tests even when a test fails.
afterEach(() => {
  cleanup();
});
