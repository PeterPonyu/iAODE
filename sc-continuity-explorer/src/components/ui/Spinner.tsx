/**
 * Loading spinner component
 */

'use client';

import { HTMLAttributes, forwardRef } from 'react';

export interface SpinnerProps extends HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg';
  label?: string;
}

export const Spinner = forwardRef<HTMLDivElement, SpinnerProps>(
  (
    {
      size = 'md',
      label = 'Loading...',
      className = '',
      ...props
    },
    ref
  ) => {
    const sizeClasses = {
      sm: 'w-4 h-4 border-2',
      md: 'w-8 h-8 border-3',
      lg: 'w-12 h-12 border-4',
    };

    return (
      <div
        ref={ref}
        className={`flex flex-col items-center justify-center gap-2 ${className}`}
        role="status"
        aria-live="polite"
        {...props}
      >
        <div
          className={`${sizeClasses[size]} border-muted-foreground border-t-transparent rounded-full animate-spin`}
        />
        {label && (
          <span className="text-sm text-muted-foreground">{label}</span>
        )}
      </div>
    );
  }
);

Spinner.displayName = 'Spinner';