import type { Metadata, Viewport } from "next";

import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: { default: "MediVision AI — Intelligent Medical Imaging Copilot", template: "%s · MediVision AI" },
  description:
    "Self-trained medical imaging copilot: chest X-ray ensemble, fracture and brain MRI models, explainability, triage and a self-built clinical language engine. Research prototype — not a medical device.",
};

export const viewport: Viewport = { themeColor: "#070B14", colorScheme: "dark light" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-dvh antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
