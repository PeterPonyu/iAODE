/**
 * Sidebar layout for controls
 */

'use client';

import { HTMLAttributes } from 'react';

export interface SidebarProps extends HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function Sidebar({ children, className = '', ...props }: SidebarProps) {
  return (
    <aside
      className={`w-full lg:w-80 lg:h-screen lg:sticky lg:top-0 overflow-y-auto ${className}`}
      {...props}
    >
      <div className="p-4 space-y-4">
        {children}
      </div>
    </aside>
  );
}