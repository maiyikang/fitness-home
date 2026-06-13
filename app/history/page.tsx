"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import BottomNavigation from "@/app/components/BottomNavigation";
import { restaurants } from "@/app/data/restaurants";
import { getHistory, type HistoryItem } from "@/app/lib/historyStorage";

interface HistoryRestaurant {
  id: string;
  name: string;
  category: string;
  viewedAt: string;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    setHistory(getHistory());
  }, []);

  const historyRestaurants: HistoryRestaurant[] = history
    .map((item) => {
      const restaurant = restaurants.find(
        (restaurantItem) => restaurantItem.id === item.restaurantId
      );

      if (!restaurant) {
        return null;
      }

      return {
        id: restaurant.id,
        name: restaurant.name,
        category: restaurant.category,
        viewedAt: item.viewedAt,
      };
    })
    .filter((restaurant): restaurant is HistoryRestaurant => {
      return restaurant !== null;
    });

  return (
    <main className="min-h-screen bg-gray-50 px-6 pb-24 pt-10">
      <section className="mx-auto max-w-3xl">
        <h1 className="text-3xl font-bold text-gray-900">History</h1>

        <p className="mt-3 text-gray-600">
          Your recently viewed restaurants will appear here.
        </p>

        <div className="mt-8 space-y-4">
          {historyRestaurants.length === 0 ? (
            <div className="rounded-2xl bg-white p-8 text-center shadow-sm">
              <p className="text-gray-600">No browsing history yet.</p>
            </div>
          ) : (
            historyRestaurants.map((restaurant) => (
              <Link
                key={restaurant.id}
                href={`/restaurants/${restaurant.id}`}
                className="block rounded-2xl bg-white p-6 shadow-sm transition active:scale-[0.98]"
              >
                <h2 className="text-xl font-semibold text-gray-900">
                  {restaurant.name}
                </h2>

                <p className="mt-1 text-sm text-gray-500">
                  {restaurant.category}
                </p>

                <p className="mt-4 text-sm text-gray-400">
                  Viewed at {new Date(restaurant.viewedAt).toLocaleString()}
                </p>
              </Link>
            ))
          )}
        </div>
      </section>

      <BottomNavigation />
    </main>
  );
}