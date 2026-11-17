import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  devIndicators: false,
  images: { unoptimized: true },
  poweredByHeader: false,
};

export default nextConfig;
