import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/iAODE',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

export default nextConfig;
