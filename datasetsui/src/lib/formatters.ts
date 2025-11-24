/**
 * Format number with commas and optional compact format
 * Example: 5748 → "5,748" or "5.7K" (compact)
 */
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
 * Example: 33.8 → "33.8 MB"
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
 * Parse numeric value from string or number
 */
export function parseNumeric(value: string | number | null | undefined): number {
  if (value === null || value === undefined || value === 'N/A') {
    return 0;
  }
  
  const num = typeof value === 'string' ? parseFloat(value) : value;
  
  return isNaN(num) ? 0 : num;
}

/**
 * Get category badge color classes using CSS custom properties
 */
export function getCategoryColor(category: string): {
  bg: string;
  text: string;
  border: string;
} {
  const colors = {
    tiny: {
      bg: 'bg-[rgb(var(--category-tiny-bg))]',
      text: 'text-[rgb(var(--category-tiny-text))]',
      border: 'border-[rgb(var(--category-tiny-border))]'
    },
    small: {
      bg: 'bg-[rgb(var(--category-small-bg))]',
      text: 'text-[rgb(var(--category-small-text))]',
      border: 'border-[rgb(var(--category-small-border))]'
    },
    medium: {
      bg: 'bg-[rgb(var(--category-medium-bg))]',
      text: 'text-[rgb(var(--category-medium-text))]',
      border: 'border-[rgb(var(--category-medium-border))]'
    },
    large: {
      bg: 'bg-[rgb(var(--category-large-bg))]',
      text: 'text-[rgb(var(--category-large-text))]',
      border: 'border-[rgb(var(--category-large-border))]'
    },
    error: {
      bg: 'bg-[rgb(var(--category-error-bg))]',
      text: 'text-[rgb(var(--category-error-text))]',
      border: 'border-[rgb(var(--category-error-border))]'
    }
  };
  
  return colors[category as keyof typeof colors] || colors.error;
}

/**
 * Get category information (label and description)
 */
export function getCategoryInfo(category: string): {
  label: string;
  description: string;
} {
  const info = {
    tiny: {
      label: 'Tiny',
      description: 'Tiny (1-5k cells)'
    },
    small: {
      label: 'Small',
      description: 'Small (5-10k cells)'
    },
    medium: {
      label: 'Medium',
      description: 'Medium (10-20k cells)'
    },
    large: {
      label: 'Large',
      description: 'Large (20k+ cells)'
    },
    error: {
      label: 'Error',
      description: 'Data parsing error'
    }
  };
  
  return info[category as keyof typeof info] || info.error;
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