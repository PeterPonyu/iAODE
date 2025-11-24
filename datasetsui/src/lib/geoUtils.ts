/**
 * Extract GSM ID from data file name
 * Example: "GSM5124060_PDX_360_Infigratinib_peak_bc_matrix.h5" → "GSM5124060"
 */
export function extractGsmId(dataFileName: string): string {
  const match = dataFileName.match(/^(GSM\d+)/);
  return match ? match[1] : '';
}

/**
 * Generate NCBI GEO download URL for GSM accession
 */
export function generateDownloadUrl(gsmId: string): string {
  if (!gsmId) return '';
  return `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${gsmId}`;
}

/**
 * Generate NCBI GEO page URL for GSE accession
 */
export function generateGeoUrl(gseAccession: string): string {
  if (!gseAccession) return '';
  return `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${gseAccession}`;
}

/**
 * Validate GSE accession format
 */
export function isValidGSE(gseAccession: string): boolean {
  return /^GSE\d+$/i.test(gseAccession);
}

/**
 * Validate GSM accession format
 */
export function isValidGSM(gsmId: string): boolean {
  return /^GSM\d+$/i.test(gsmId);
}