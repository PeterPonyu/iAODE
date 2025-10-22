/**
 * Debounce hook for slider and input controls
 */

import { useEffect, useState } from 'react';

/**
 * Debounce a value - delays updates until user stops changing it
 * 
 * @param value - The value to debounce
 * @param delay - Delay in milliseconds (default: 300ms)
 * @returns Debounced value
 * 
 * @example
 * const [continuity, setContinuity] = useState(0.95);
 * const debouncedContinuity = useDebounce(continuity, 300);
 * 
 * // continuity updates immediately (for UI responsiveness)
 * // debouncedContinuity updates after 300ms of no changes (for API calls)
 */
export function useDebounce<T>(value: T, delay: number = 300): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    // Set up the timeout
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    // Cleanup: cancel the timeout if value changes before delay expires
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * Debounce a callback function
 * 
 * @param callback - Function to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced callback function
 * 
 * @example
 * const handleSearch = useDebouncedCallback((query: string) => {
 *   // This only runs after user stops typing for 300ms
 *   fetchSearchResults(query);
 * }, 300);
 */
export function useDebouncedCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number = 300
): (...args: Parameters<T>) => void {
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [timeoutId]);

  return (...args: Parameters<T>) => {
    // Clear previous timeout
    if (timeoutId) clearTimeout(timeoutId);

    // Set new timeout
    const newTimeoutId = setTimeout(() => {
      callback(...args);
    }, delay);

    setTimeoutId(newTimeoutId);
  };
}