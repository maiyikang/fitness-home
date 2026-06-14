import type { Restaurant } from "@/app/types/restaurant";

const STORAGE_KEY = "realRestaurants";

export function saveRealRestaurants(restaurants: Restaurant[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(restaurants));
}

export function getRealRestaurants(): Restaurant[] {
  const storedValue = localStorage.getItem(STORAGE_KEY);

  if (!storedValue) {
    return [];
  }

  try {
    return JSON.parse(storedValue) as Restaurant[];
  } catch {
    return [];
  }
}

export function getRealRestaurantById(id: string): Restaurant | null {
  const restaurants = getRealRestaurants();

  return restaurants.find((restaurant) => restaurant.id === id) ?? null;
}