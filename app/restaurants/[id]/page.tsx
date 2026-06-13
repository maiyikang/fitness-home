"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { useEffect, useState } from "react";

import FavoriteButton from "@/app/components/FavoriteButton";
import HistoryTracker from "@/app/components/HistoryTracker";
import { restaurants } from "@/app/data/restaurants";
import { profile as defaultProfile } from "@/app/data/profile";
import { getStoredProfile } from "@/app/lib/profileStorage";
import {
  calculateMatchScore,
  getMatchScoreReasons,
} from "@/app/lib/scoring";
import type { UserProfile } from "@/app/types/profile";

export default function RestaurantDetailPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;

  const [userProfile, setUserProfile] =
    useState<UserProfile>(defaultProfile);

  useEffect(() => {
    setUserProfile(getStoredProfile());
  }, []);

  const restaurant = restaurants.find((item) => item.id === id);

  if (!restaurant) {
    notFound();
  }

  const calculatedScore = calculateMatchScore(restaurant, userProfile);
  const scoreReasons = getMatchScoreReasons(restaurant, userProfile);

  return (
    <>
      <HistoryTracker restaurantId={restaurant.id} />

      <main className="min-h-screen bg-gray-50 px-5 pb-24 pt-6">
        <Link
          href="/restaurants"
          className="text-sm font-medium text-green-600"
        >
          ← Back to Restaurants
        </Link>

        <section className="mt-4 rounded-3xl bg-white p-6 shadow-sm">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                {restaurant.name}
              </h1>

              <p className="mt-2 text-gray-500">
                {restaurant.category}
              </p>
            </div>

            <FavoriteButton restaurantId={restaurant.id} />
          </div>

          <div className="mt-6 grid grid-cols-3 gap-3">
            <div className="rounded-2xl bg-gray-50 p-3 text-center">
              <p className="text-xs text-gray-400">Calories</p>
              <p className="mt-1 text-lg font-bold">
                {restaurant.averageCalories}
              </p>
            </div>

            <div className="rounded-2xl bg-gray-50 p-3 text-center">
              <p className="text-xs text-gray-400">Protein</p>
              <p className="mt-1 text-lg font-bold">
                {restaurant.proteinScore}
              </p>
            </div>

            <div className="rounded-2xl bg-gray-50 p-3 text-center">
              <p className="text-xs text-gray-400">Health</p>
              <p className="mt-1 text-lg font-bold">
                {restaurant.healthScore}
              </p>
            </div>
          </div>

          <div className="mt-6">
            <p className="mb-2 text-sm font-semibold text-gray-500">
              Match Score
            </p>

            <div className="rounded-2xl bg-green-100 p-4 text-center">
              <p className="text-4xl font-bold text-green-700">
                {calculatedScore}
              </p>
            </div>
          </div>

          <div className="mt-6">
            <h2 className="text-lg font-bold text-gray-900">
              Why this score?
            </h2>

            <div className="mt-3 space-y-3">
              {scoreReasons.map((reason) => (
                <div
                  key={`${reason.label}-${reason.points}`}
                  className="flex items-center justify-between rounded-2xl bg-gray-50 px-4 py-3"
                >
                  <p className="text-sm font-medium text-gray-700">
                    {reason.label}
                  </p>

                  <p
                    className={
                      reason.type === "negative"
                        ? "text-sm font-bold text-red-500"
                        : reason.type === "positive"
                          ? "text-sm font-bold text-green-600"
                          : "text-sm font-bold text-gray-500"
                    }
                  >
                    {reason.points > 0
                      ? `+${reason.points}`
                      : reason.points}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            {restaurant.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full bg-green-100 px-3 py-1 text-sm font-medium text-green-700"
              >
                {tag}
              </span>
            ))}
          </div>
        </section>
      </main>
    </>
  );
}