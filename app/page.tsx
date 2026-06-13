"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import BottomNavigation from "@/app/components/BottomNavigation";
import FavoriteButton from "@/app/components/FavoriteButton";

import { restaurants } from "@/app/data/restaurants";
import { profile as defaultProfile } from "@/app/data/profile";
import { calculateMatchScore } from "@/app/lib/scoring";
import type { UserProfile } from "@/app/types/profile";

export default function RestaurantsPage() {
  const [userProfile, setUserProfile] =
    useState<UserProfile>(defaultProfile);

  useEffect(() => {
    const savedProfile = localStorage.getItem("userProfile");

    if (savedProfile) {
      setUserProfile(JSON.parse(savedProfile) as UserProfile);
    }
  }, []);

  const sortedRestaurants = [...restaurants]
    .map((restaurant) => ({
      ...restaurant,
      calculatedScore: calculateMatchScore(restaurant, userProfile),
    }))
    .sort((a, b) => b.calculatedScore - a.calculatedScore);

  return (
    <main className="min-h-screen bg-gray-50 px-5 pb-24 pt-6">
      <section className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Restaurants</h1>

        <p className="mt-2 text-sm text-gray-500">
          Ranked by your fitness goals, calories, protein, and meal preference.
        </p>
      </section>

      <section className="space-y-4">
        {sortedRestaurants.map((restaurant) => (
          <div
            key={restaurant.id}
            className="relative rounded-3xl bg-white p-5 shadow-sm"
          >
            <div className="absolute right-6 top-6 z-10">
              <FavoriteButton restaurantId={restaurant.id} />
            </div>

            <Link
              href={`/restaurants/${restaurant.id}`}
              className="block pr-14 transition active:scale-[0.98]"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900">
                    {restaurant.name}
                  </h2>

                  <p className="mt-1 text-sm font-medium text-gray-500">
                    {restaurant.category}
                  </p>
                </div>

                <div className="mr-12 rounded-2xl bg-green-100 px-3 py-2 text-center">
                  <p className="text-lg font-bold text-green-700">
                    {restaurant.calculatedScore}
                  </p>

                  <p className="text-xs font-medium text-green-700">Match</p>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-3">
                <div className="rounded-2xl bg-gray-50 p-3">
                  <p className="text-xs text-gray-400">Calories</p>
                  <p className="mt-1 font-bold text-gray-900">
                    {restaurant.averageCalories}
                  </p>
                </div>

                <div className="rounded-2xl bg-gray-50 p-3">
                  <p className="text-xs text-gray-400">Protein</p>
                  <p className="mt-1 font-bold text-gray-900">
                    {restaurant.proteinScore}
                  </p>
                </div>

                <div className="rounded-2xl bg-gray-50 p-3">
                  <p className="text-xs text-gray-400">Health</p>
                  <p className="mt-1 font-bold text-gray-900">
                    {restaurant.healthScore}
                  </p>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {restaurant.tags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full bg-gray-100 px-3 py-1 text-xs font-semibold text-gray-600"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </Link>
          </div>
        ))}
      </section>

      <BottomNavigation />
    </main>
  );
}