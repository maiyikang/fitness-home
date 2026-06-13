"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import BottomNavigation from "@/app/components/BottomNavigation";

import { restaurants } from "@/app/data/restaurants";

import { profile as defaultProfile } from "@/app/data/profile";

import { calculateMatchScore } from "@/app/lib/scoring";
import { getStoredProfile } from "@/app/lib/profileStorage";

import type { UserProfile } from "@/app/types/profile";

export default function FavoritesPage() {
  const [favoriteIds, setFavoriteIds] = useState<string[]>([]);
  const [userProfile, setUserProfile] =
    useState<UserProfile>(defaultProfile);

  useEffect(() => {
    const savedFavorites =
      localStorage.getItem("favoriteRestaurants");

    const ids = savedFavorites
      ? (JSON.parse(savedFavorites) as string[])
      : [];

    setFavoriteIds(ids);

    setUserProfile(getStoredProfile());
  }, []);

  const favoriteRestaurants = restaurants
    .filter((restaurant) =>
      favoriteIds.includes(restaurant.id)
    )
    .map((restaurant) => ({
      ...restaurant,
      calculatedScore: calculateMatchScore(
        restaurant,
        userProfile
      ),
    }))
    .sort(
      (a, b) =>
        b.calculatedScore - a.calculatedScore
    );

  return (
    <main className="min-h-screen bg-gray-50 px-6 pb-24 pt-10">
      <section className="mx-auto max-w-3xl">
        <h1 className="text-3xl font-bold text-gray-900">
          Favorites
        </h1>

        <p className="mt-3 text-gray-600">
          Your saved restaurants will appear here.
        </p>

        <div className="mt-8 space-y-4">
          {favoriteRestaurants.length === 0 ? (
            <div className="rounded-2xl bg-white p-8 text-center shadow-sm">
              <p className="text-gray-600">
                No favorite restaurants yet.
              </p>
            </div>
          ) : (
            favoriteRestaurants.map((restaurant) => (
              <Link
                key={restaurant.id}
                href={`/restaurants/${restaurant.id}`}
                className="block rounded-2xl bg-white p-6 shadow-sm transition active:scale-[0.98]"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900">
                      {restaurant.name}
                    </h2>

                    <p className="mt-1 text-sm text-gray-500">
                      {restaurant.category}
                    </p>
                  </div>

                  <div className="rounded-full bg-green-100 px-3 py-1 text-sm font-medium text-green-700">
                    {restaurant.calculatedScore}% Match
                  </div>
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
                  <div className="rounded-xl bg-gray-50 p-3">
                    <p className="text-gray-500">
                      Calories
                    </p>

                    <p className="mt-1 font-semibold text-gray-900">
                      {restaurant.averageCalories} kcal
                    </p>
                  </div>

                  <div className="rounded-xl bg-gray-50 p-3">
                    <p className="text-gray-500">
                      Protein
                    </p>

                    <p className="mt-1 font-semibold text-gray-900">
                      {restaurant.proteinScore}/100
                    </p>
                  </div>

                  <div className="rounded-xl bg-gray-50 p-3">
                    <p className="text-gray-500">
                      Health
                    </p>

                    <p className="mt-1 font-semibold text-gray-900">
                      {restaurant.healthScore}/100
                    </p>
                  </div>

                  <div className="rounded-xl bg-gray-50 p-3">
                    <p className="text-gray-500">
                      Type
                    </p>

                    <p className="mt-1 font-semibold text-gray-900">
                      {restaurant.category}
                    </p>
                  </div>
                </div>
              </Link>
            ))
          )}
        </div>
      </section>

      <BottomNavigation />
    </main>
  );
}