/**
 * Alert component for messages, errors, and warnings
 */

'use client';

import { HTMLAttributes, forwardRef } from 'react';

export interface AlertProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'info' | 'success' | 'warning' | 'error';
  title?: string;
  onClose?: () => void;
}

export const Alert = forwardRef<HTMLDivElement, AlertProps>(
  (
    {
      variant = 'info',
      title,
      onClose,
      className = '',
      children,
      ...props
    },
    ref
  ) => {
    const variantStyles = {
      info: {
        backgroundColor: 'var(--color-primary-50)',
        borderColor: 'var(--color-primary-300)',
        color: 'var(--color-primary-900)',
      },
      success: {
        backgroundColor: '#dcfce7',
        borderColor: '#86efac',
        color: '#166534',
      },
      warning: {
        backgroundColor: '#fef3c7',
        borderColor: '#fcd34d',
        color: '#92400e',
      },
      error: {
        backgroundColor: '#fee2e2',
        borderColor: '#fca5a5',
        color: '#991b1b',
      },
    };

    const icons = {
      info: 'ℹ️',
      success: '✅',
      warning: '⚠️',
      error: '❌',
    };

    return (
      <div
        ref={ref}
        className={`rounded-lg border-2 p-4 ${className}`}
        style={variantStyles[variant]}
        role="alert"
        {...props}
      >
        <div className="flex items-start gap-3">
          <span className="text-xl">{icons[variant]}</span>
          
          <div className="flex-1">
            {title && (
              <h4 className="font-semibold mb-1">{title}</h4>
            )}
            {children && (
              <div className="text-sm">{children}</div>
            )}
          </div>
          
          {onClose && (
            <button
              onClick={onClose}
              className="text-current opacity-70 hover:opacity-100 transition-opacity"
              aria-label="Close alert"
            >
              ✕
            </button>
          )}
        </div>
      </div>
    );
  }
);

Alert.displayName = 'Alert';