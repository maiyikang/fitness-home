export function getFavorites(): string[] {
  if (typeof window === "undefined") {
    return [];
  }

  const saved = localStorage.getItem("favorites");

  if (!saved) {
    return [];
  }

  return JSON.parse(saved);
}

export function isFavorite(id: string): boolean {
  return getFavorites().includes(id);
}

export function toggleFavorite(id: string) {
  const favorites = getFavorites();

  const exists = favorites.includes(id);

  const updated = exists
    ? favorites.filter((item) => item !== id)
    : [...favorites, id];

  localStorage.setItem(
    "favorites",
    JSON.stringify(updated)
  );

  return updated;
}