/**
 * Slider (range input) component
 */

'use client';

import { forwardRef, InputHTMLAttributes, useState, useId } from 'react';

export interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  showValue?: boolean;
  formatValue?: (value: number) => string;
}

export const Slider = forwardRef<HTMLInputElement, SliderProps>(
  (
    {
      label,
      showValue = true,
      formatValue = (v) => v.toString(),
      className = '',
      id,
      min = 0,
      max = 100,
      step = 1,
      value: controlledValue,
      defaultValue,
      onChange,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const sliderId = id || generatedId;
    
    // Handle both controlled and uncontrolled
    const [internalValue, setInternalValue] = useState(
      defaultValue !== undefined ? Number(defaultValue) : Number(min)
    );
    
    const value = controlledValue !== undefined ? Number(controlledValue) : internalValue;

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = Number(e.target.value);
      if (controlledValue === undefined) {
        setInternalValue(newValue);
      }
      onChange?.(e);
    };

    return (
      <div className="space-y-2">
        {(label || showValue) && (
          <div className="flex items-center justify-between">
            {label && (
              <label htmlFor={sliderId} className="text-sm font-medium">
                {label}
              </label>
            )}
            {showValue && (
              <span className="text-sm font-mono text-muted-foreground">
                {formatValue(value)}
              </span>
            )}
          </div>
        )}
        
        <input
          ref={ref}
          type="range"
          id={sliderId}
          className={`slider ${className}`}
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          {...props}
        />
        
        {/* Range indicators */}
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{formatValue(Number(min))}</span>
          <span>{formatValue(Number(max))}</span>
        </div>
      </div>
    );
  }
);

Slider.displayName = 'Slider';