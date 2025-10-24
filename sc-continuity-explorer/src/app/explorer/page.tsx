// ============================================================================
// FILE: app/explorer/page.tsx
// Explorer router page
// ============================================================================

import { ExplorerView } from '@/components/ExplorerView';

export const metadata = {
  title: 'Explorer - Single-Cell Continuity',
  description: 'Interactive exploration of single-cell trajectory data',
};

export default function ExplorerPage() {
  return <ExplorerView />;
}