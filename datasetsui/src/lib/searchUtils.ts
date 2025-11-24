import Fuse from 'fuse.js';
import { GSEGroup } from '@/types/datasets';

export function createSearchIndex(gseGroups: GSEGroup[]) {
  return new Fuse(gseGroups, {
    keys: [
      { name: 'gseAccession', weight: 2 },
      { name: 'title', weight: 1.5 },
      { name: 'authors', weight: 1 },
      { name: 'organism', weight: 0.8 },
    ],
    threshold: 0.4,
    includeScore: true,
  });
}

export function searchGSE(
  gseGroups: GSEGroup[],
  query: string
): GSEGroup[] {
  if (!query.trim()) return gseGroups;
  
  const fuse = createSearchIndex(gseGroups);
  const results = fuse.search(query);
  return results.map(r => r.item);
}