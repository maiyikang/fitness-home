"use client";

import { useEffect } from "react";

import { addToHistory } from "@/app/lib/historyStorage";

interface HistoryTrackerProps {
  restaurantId: string;
}

export default function HistoryTracker({
  restaurantId,
}: HistoryTrackerProps) {
  useEffect(() => {
    addToHistory(restaurantId);
  }, [restaurantId]);

  return null;
}