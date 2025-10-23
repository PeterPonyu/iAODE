/**
 * Wrapper component to prevent SSR hydration issues
 * Only renders children on client-side after mount
 */

'use client';

import { useEffect, useState } from 'react';

interface ClientOnlyProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

/**
 * Renders children only on client-side (after hydration)
 * 
 * @example
 * <ClientOnly fallback={<div>Loading...</div>}>
 *   <ThemeSwitcher />
 * </ClientOnly>
 */
export function ClientOnly({ children, fallback = null }: ClientOnlyProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
}