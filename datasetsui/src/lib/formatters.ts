// lib/formatters.ts

/**
 * Format number with commas
 * Example: 5748 → "5,748"
 */
// Add compact format option for large numbers
export function formatNumber(
  value: string | number | null | undefined,
  compact = false
): string {
  if (value === null || value === undefined || value === 'N/A') {
    return 'N/A';
  }
  
  const num = typeof value === 'string' ? parseFloat(value) : value;
  
  if (isNaN(num)) {
    return 'N/A';
  }
  
  // Compact format for large numbers (e.g., "3.2M", "125K")
  if (compact) {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  }
  
  return num.toLocaleString('en-US');
}

/**
 * Format file size with unit
 * Example: "33.8" → "33.8 MB"
 */
export function formatFileSize(sizeMB: string | number | null | undefined): string {
  if (sizeMB === null || sizeMB === undefined || sizeMB === 'N/A') {
    return 'N/A';
  }
  
  const size = typeof sizeMB === 'string' ? parseFloat(sizeMB) : sizeMB;
  
  if (isNaN(size)) {
    return 'N/A';
  }
  
  if (size >= 1024) {
    return `${(size / 1024).toFixed(2)} GB`;
  }
  
  return `${size.toFixed(2)} MB`;
}

/**
 * Parse numeric value from string
 */
export function parseNumeric(value: string | number | null | undefined): number {
  if (value === null || value === undefined || value === 'N/A') {
    return 0;
  }
  
  const num = typeof value === 'string' ? parseFloat(value) : value;
  
  return isNaN(num) ? 0 : num;
}

/**
 * Get category badge color classes
 */
export function getCategoryColor(category: string): {
  bg: string;
  text: string;
  border: string;
} {
  const colors = {
    tiny: {
      bg: 'bg-gray-100 dark:bg-gray-800',
      text: 'text-gray-700 dark:text-gray-300',
      border: 'border-gray-300 dark:border-gray-600'
    },
    small: {
      bg: 'bg-blue-100 dark:bg-blue-900/30',
      text: 'text-blue-700 dark:text-blue-300',
      border: 'border-blue-300 dark:border-blue-600'
    },
    medium: {
      bg: 'bg-yellow-100 dark:bg-yellow-900/30',
      text: 'text-yellow-700 dark:text-yellow-300',
      border: 'border-yellow-300 dark:border-yellow-600'
    },
    large: {
      bg: 'bg-green-100 dark:bg-green-900/30',
      text: 'text-green-700 dark:text-green-300',
      border: 'border-green-300 dark:border-green-600'
    },
    error: {
      bg: 'bg-red-100 dark:bg-red-900/30',
      text: 'text-red-700 dark:text-red-300',
      border: 'border-red-300 dark:border-red-600'
    }
  };
  
  return colors[category as keyof typeof colors] || colors.error;
}

/**
 * Truncate text with ellipsis
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
}

/**
 * Capitalize first letter
 */
export function capitalize(text: string): string {
  if (!text) return '';
  return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
}