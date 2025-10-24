/**
 * Badge/Label component for tags and status indicators
 */

'use client';

import { HTMLAttributes, forwardRef } from 'react';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md';
}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  (
    {
      variant = 'default',
      size = 'md',
      className = '',
      children,
      ...props
    },
    ref
  ) => {
    const baseClasses = 'inline-flex items-center font-medium rounded-full border';
    
    const sizeClasses = {
      sm: 'px-2 py-0.5 text-xs',
      md: 'px-2.5 py-0.5 text-sm',
    };

    const variantStyles = {
      default: {
        backgroundColor: 'var(--color-muted)',
        color: 'var(--color-muted-foreground)',
        borderColor: 'var(--color-border)',
      },
      primary: {
        backgroundColor: 'var(--color-primary-100)',
        color: 'var(--color-primary-900)',
        borderColor: 'var(--color-primary-300)',
      },
      success: {
        backgroundColor: '#dcfce7',
        color: '#166534',
        borderColor: '#86efac',
      },
      warning: {
        backgroundColor: '#fef3c7',
        color: '#92400e',
        borderColor: '#fcd34d',
      },
      danger: {
        backgroundColor: '#fee2e2',
        color: '#991b1b',
        borderColor: '#fca5a5',
      },
    };

    const classes = [baseClasses, sizeClasses[size], className]
      .filter(Boolean)
      .join(' ');

    return (
      <span
        ref={ref}
        className={classes}
        style={variantStyles[variant]}
        {...props}
      >
        {children}
      </span>
    );
  }
);

Badge.displayName = 'Badge';