export interface HistoryItem {
  restaurantId: string;
  viewedAt: string;
}

const STORAGE_KEY = "restaurantHistory";

export function getHistory(): HistoryItem[] {
  if (typeof window === "undefined") {
    return [];
  }

  const savedHistory = localStorage.getItem(STORAGE_KEY);

  if (!savedHistory) {
    return [];
  }

  return JSON.parse(savedHistory) as HistoryItem[];
}

export function addToHistory(restaurantId: string) {
  const currentHistory = getHistory();

  const nextHistory: HistoryItem[] = [
    {
      restaurantId,
      viewedAt: new Date().toISOString(),
    },
    ...currentHistory.filter((item) => item.restaurantId !== restaurantId),
  ].slice(0, 20);

  localStorage.setItem(STORAGE_KEY, JSON.stringify(nextHistory));
}