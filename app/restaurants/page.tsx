"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import BottomNavigation from "@/app/components/BottomNavigation";
import FavoriteButton from "@/app/components/FavoriteButton";
import { restaurants } from "@/app/data/restaurants";
import { profile as defaultProfile } from "@/app/data/profile";
import { getStoredProfile } from "@/app/lib/profileStorage";
import {
  calculateMatchScore,
  getMatchScoreReasons,
} from "@/app/lib/scoring";
import type { UserProfile } from "@/app/types/profile";

export default function RestaurantsPage() {
  const [userProfile, setUserProfile] =
    useState<UserProfile>(defaultProfile);

  useEffect(() => {
    setUserProfile(getStoredProfile());
  }, []);

  const sortedRestaurants = [...restaurants].sort((a, b) => {
    return (
      calculateMatchScore(b, userProfile) -
      calculateMatchScore(a, userProfile)
    );
  });

  return (
    <>
      <main className="min-h-screen bg-gray-50 px-5 pb-24 pt-6">
        <section className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Restaurants
          </h1>

          <p className="mt-2 text-sm text-gray-500">
            Ranked by your fitness goals, calories, protein, and meal preference.
          </p>
        </section>

        <section className="space-y-4">
          {sortedRestaurants.map((restaurant) => {
            const calculatedScore = calculateMatchScore(
              restaurant,
              userProfile
            );

            const topReasons = getMatchScoreReasons(
              restaurant,
              userProfile
            )
              .filter((reason) => reason.type === "positive")
              .slice(0, 2);

            return (
              <Link
                key={restaurant.id}
                href={`/restaurants/${restaurant.id}`}
                className="block rounded-3xl bg-white p-5 shadow-sm transition active:scale-[0.98]"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-xl font-bold text-gray-900">
                      {restaurant.name}
                    </h2>

                    <p className="mt-1 text-sm text-gray-500">
                      {restaurant.category}
                    </p>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="rounded-2xl bg-green-100 px-4 py-2 text-center">
                      <p className="text-xs font-medium text-green-700">
                        Score
                      </p>

                      <p className="text-xl font-bold text-green-700">
                        {calculatedScore}
                      </p>
                    </div>

                    <FavoriteButton restaurantId={restaurant.id} />
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-3 gap-3">
                  <div className="rounded-2xl bg-gray-50 p-3 text-center">
                    <p className="text-xs text-gray-400">Calories</p>
                    <p className="mt-1 text-sm font-bold">
                      {restaurant.averageCalories}
                    </p>
                  </div>

                  <div className="rounded-2xl bg-gray-50 p-3 text-center">
                    <p className="text-xs text-gray-400">Protein</p>
                    <p className="mt-1 text-sm font-bold">
                      {restaurant.proteinScore}
                    </p>
                  </div>

                  <div className="rounded-2xl bg-gray-50 p-3 text-center">
                    <p className="text-xs text-gray-400">Health</p>
                    <p className="mt-1 text-sm font-bold">
                      {restaurant.healthScore}
                    </p>
                  </div>
                </div>

                {topReasons.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {topReasons.map((reason) => (
                      <p
                        key={`${restaurant.id}-${reason.label}`}
                        className="text-sm font-medium text-green-700"
                      >
                        ✓ {reason.label}
                      </p>
                    ))}
                  </div>
                )}

                <div className="mt-4 flex flex-wrap gap-2">
                  {restaurant.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </Link>
            );
          })}
        </section>
      </main>

      <BottomNavigation />
    </>
  );
}