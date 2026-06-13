"use client";

import { useEffect, useState } from "react";

interface FavoriteButtonProps {
  restaurantId: string;
}

const STORAGE_KEY = "favoriteRestaurants";

export default function FavoriteButton({
  restaurantId,
}: FavoriteButtonProps) {
  const [isFavorite, setIsFavorite] = useState(false);

  useEffect(() => {
    const savedFavorites = localStorage.getItem(STORAGE_KEY);
    const favoriteIds = savedFavorites
      ? (JSON.parse(savedFavorites) as string[])
      : [];

    setIsFavorite(favoriteIds.includes(restaurantId));
  }, [restaurantId]);

  function handleClick(event: React.MouseEvent<HTMLButtonElement>) {
    event.preventDefault();
    event.stopPropagation();

    const savedFavorites = localStorage.getItem(STORAGE_KEY);
    const favoriteIds = savedFavorites
      ? (JSON.parse(savedFavorites) as string[])
      : [];

    const nextFavoriteIds = favoriteIds.includes(restaurantId)
      ? favoriteIds.filter((id) => id !== restaurantId)
      : [...favoriteIds, restaurantId];

    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextFavoriteIds));
    setIsFavorite(nextFavoriteIds.includes(restaurantId));
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className="text-2xl transition active:scale-90"
      aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
    >
      {isFavorite ? "❤️" : "🤍"}
    </button>
  );
}