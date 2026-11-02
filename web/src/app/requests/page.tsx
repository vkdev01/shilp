import type { Metadata } from "next";

import { Requests } from "@/components/requests";

export const metadata: Metadata = { title: "Requests" };

export default function Page() {
  return <Requests />;
}
