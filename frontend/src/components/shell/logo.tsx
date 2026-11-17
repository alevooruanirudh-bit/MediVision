import { cn } from "@/lib/utils";

export function Logo({ className, withText = true }: { className?: string; withText?: boolean }) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
        <defs>
          <linearGradient id="mvg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#22D3EE" />
            <stop offset="1" stopColor="#8B5CF6" />
          </linearGradient>
        </defs>
        <rect x="1" y="1" width="30" height="30" rx="9" fill="url(#mvg)" opacity="0.16" />
        <rect x="1" y="1" width="30" height="30" rx="9" fill="none" stroke="url(#mvg)" strokeWidth="1.2" />
        <path d="M7 17.5h4l2.2-6 3.6 11 2.6-7.5 1.6 2.5H25" fill="none" stroke="url(#mvg)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {withText && (
        <span className="text-[15px] font-semibold tracking-tight text-fg">
          MediVision<span className="text-gradient"> AI</span>
        </span>
      )}
    </span>
  );
}
