import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/iAODE/explorer',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

export default nextConfig;
