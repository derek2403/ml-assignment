/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'cdn0.iconfinder.com',
        pathname: '/data/icons/**',
      },
    ],
  },
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://e3c329acf714051138becd9199470e6d1ae0cabd-5050.dstack-prod5.phala.network:5050/:path*',
      },
    ];
  },
};

export default nextConfig;
