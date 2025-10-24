/**
 * Main content area wrapper
 */

'use client';

import { HTMLAttributes } from 'react';

export interface MainContentProps extends HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function MainContent({ children, className = '', ...props }: MainContentProps) {
  return (
    <main
      className={`flex-1 p-4 md:p-6 lg:p-8 ${className}`}
      {...props}
    >
      {children}
    </main>
  );
}