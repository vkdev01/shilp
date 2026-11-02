import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-[70dvh] flex-col items-center justify-center gap-3 text-center">
      <p className="font-mono text-sm text-accent">404</p>
      <h1 className="text-2xl font-semibold tracking-tight">Nothing served here</h1>
      <p className="max-w-sm text-sm text-muted">
        This path is not a page or an API route. The API lives under <code>/api</code> and <code>/v1</code>.
      </p>
      <Link href="/" className="mt-2 text-sm font-medium text-accent hover:underline">
        Back to the overview →
      </Link>
    </div>
  );
}
