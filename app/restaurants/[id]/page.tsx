"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import FavoriteButton from "@/app/components/FavoriteButton";
import HistoryTracker from "@/app/components/HistoryTracker";
import { restaurants } from "@/app/data/restaurants";
import { profile as defaultProfile } from "@/app/data/profile";
import { getMealRecommendation } from "@/app/lib/mealRecommendation";
import { getRecommendedMenuItems } from "@/app/lib/menuRecommendation";
import { getStoredProfile } from "@/app/lib/profileStorage";
import { getRealRestaurantById } from "@/app/lib/realRestaurantsStorage";
import {
  calculateMatchScore,
  getMatchScoreReasons,
} from "@/app/lib/scoring";
import type { UserProfile } from "@/app/types/profile";
import type { Restaurant } from "@/app/types/restaurant";

export default function RestaurantDetailPage() {
  const params = useParams<{ id: string }>();
  const id = decodeURIComponent(params.id);

  const [userProfile, setUserProfile] =
    useState<UserProfile>(defaultProfile);

  const [restaurant, setRestaurant] = useState<Restaurant | null>(null);

  useEffect(() => {
    setUserProfile(getStoredProfile());

    const sampleRestaurant =
      restaurants.find((item) => item.id === id) ?? null;

    const storedRealRestaurant = getRealRestaurantById(id);

    setRestaurant(sampleRestaurant ?? storedRealRestaurant);
  }, [id]);

  if (!restaurant) {
    return (
      <main className="min-h-screen bg-gray-50 px-5 pb-24 pt-24">
        <div className="fixed left-4 top-4 z-50">
          <Link
            href="/restaurants"
            className="flex h-12 w-12 items-center justify-center rounded-full bg-white text-xl font-bold text-green-700 shadow-lg ring-1 ring-gray-200 transition active:scale-95"
            aria-label="Back to restaurants"
          >
            ←
          </Link>
        </div>

        <section className="mt-4 rounded-3xl bg-white p-6 shadow-sm">
          <h1 className="text-2xl font-bold text-gray-900">
            Restaurant not found
          </h1>

          <p className="mt-3 text-sm text-gray-500">
            Please go back and load real restaurants again.
          </p>
        </section>
      </main>
    );
  }

  const calculatedScore = calculateMatchScore(restaurant, userProfile);
  const scoreReasons = getMatchScoreReasons(restaurant, userProfile);
  const mealRecommendation = getMealRecommendation(restaurant, userProfile);
  const recommendedMenuItems = getRecommendedMenuItems(
    restaurant,
    userProfile
  );

  return (
    <>
      <HistoryTracker restaurantId={restaurant.id} />

      <main className="min-h-screen bg-gray-50 px-5 pb-24 pt-24">
        <div className="fixed left-4 top-4 z-50">
          <Link
            href="/restaurants"
            className="flex h-12 w-12 items-center justify-center rounded-full bg-white text-xl font-bold text-green-700 shadow-lg ring-1 ring-gray-200 transition active:scale-95"
            aria-label="Back to restaurants"
          >
            ←
          </Link>
        </div>

        <section className="mt-4 rounded-3xl bg-white p-6 shadow-sm">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                {restaurant.name}
              </h1>

              <p className="mt-2 text-gray-500">
                {restaurant.category}
              </p>

              <p className="mt-1 text-sm font-medium text-gray-400">
                {restaurant.cuisine}
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

          <div className="mt-6 rounded-3xl bg-green-50 p-5">
            <h2 className="text-lg font-bold text-green-900">
              {mealRecommendation.title}
            </h2>

            <p className="mt-2 text-sm text-green-700">
              {mealRecommendation.message}
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded-2xl bg-white p-4">
                <p className="text-sm text-gray-500">Estimated Calories</p>
                <p className="mt-1 text-xl font-bold text-gray-900">
                  {mealRecommendation.calories} kcal
                </p>
              </div>

              <div className="rounded-2xl bg-white p-4">
                <p className="text-sm text-gray-500">Estimated Protein</p>
                <p className="mt-1 text-xl font-bold text-gray-900">
                  {mealRecommendation.protein} g
                </p>
              </div>
            </div>

            <div className="mt-4 space-y-2">
              {mealRecommendation.reasons.map((reason) => (
                <p
                  key={reason}
                  className="text-sm font-medium text-green-800"
                >
                  ✓ {reason}
                </p>
              ))}
            </div>
          </div>

          <div className="mt-6 rounded-3xl bg-white p-5 ring-1 ring-gray-100">
            <h2 className="text-lg font-bold text-gray-900">
              Recommended Order
            </h2>

            <p className="mt-2 text-sm text-gray-500">
              Based on your goal, meal type, and nutrition targets.
            </p>

            <div className="mt-4 space-y-4">
              {recommendedMenuItems.map((item) => (
                <div
                  key={item.id}
                  className="rounded-2xl bg-green-50 p-4"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="font-bold text-green-900">
                        {item.name}
                      </h3>

                      <p className="mt-1 text-xs font-medium text-green-700">
                        Fitness Home Score
                      </p>
                    </div>

                    <div className="rounded-2xl bg-white px-4 py-2 text-center">
                      <p className="text-xs font-medium text-green-700">
                        Score
                      </p>

                      <p className="text-2xl font-bold text-green-800">
                        {item.recommendationScore}
                      </p>
                    </div>
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-5">
                    <div>
                      <p className="text-xs text-green-700">Calories</p>
                      <p className="font-bold text-green-900">
                        {item.calories}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs text-green-700">Protein</p>
                      <p className="font-bold text-green-900">
                        {item.protein} g
                      </p>
                    </div>

                    <div>
                      <p className="text-xs text-green-700">Carbs</p>
                      <p className="font-bold text-green-900">
                        {item.carbs} g
                      </p>
                    </div>

                    <div>
                      <p className="text-xs text-green-700">Fat</p>
                      <p className="font-bold text-green-900">
                        {item.fat} g
                      </p>
                    </div>

                    <div>
                      <p className="text-xs text-green-700">Fiber</p>
                      <p className="font-bold text-green-900">
                        {item.fiber} g
                      </p>
                    </div>
                  </div>

                  <div className="mt-4 rounded-2xl bg-white p-3">
                    <p className="text-sm font-bold text-gray-900">
                      Score Breakdown
                    </p>

                    <div className="mt-3 space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Nutri Quality</span>
                        <span className="font-bold text-green-700">
                          {item.scoreBreakdown.nutriQuality}/40
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Goal Match</span>
                        <span className="font-bold text-green-700">
                          {item.scoreBreakdown.goalMatch}/30
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Calories Match</span>
                        <span className="font-bold text-green-700">
                          {item.scoreBreakdown.caloriesMatch}/20
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Macro Match</span>
                        <span className="font-bold text-green-700">
                          {item.scoreBreakdown.macroMatch}/10
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 rounded-2xl bg-white p-3">
                    <p className="text-sm font-bold text-gray-900">
                      Why recommended?
                    </p>

                    <div className="mt-2 space-y-1">
                      {item.recommendationReasons.map((reason) => (
                        <p
                          key={`${item.id}-${reason}`}
                          className="text-sm text-green-700"
                        >
                          ✓ {reason}
                        </p>
                      ))}
                    </div>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2">
                    {item.tags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded-full bg-white px-3 py-1 text-xs font-medium text-green-700"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
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

          <div className="mt-6">
            <h2 className="text-lg font-bold text-gray-900">
              Full Menu
            </h2>

            <div className="mt-3 space-y-3">
              {restaurant.menu.map((item) => (
                <div
                  key={item.id}
                  className="rounded-2xl bg-gray-50 p-4"
                >
                  <h3 className="font-bold text-gray-900">
                    {item.name}
                  </h3>

                  <p className="mt-1 text-sm text-gray-500">
                    {item.calories} kcal · {item.protein}g protein ·{" "}
                    {item.carbs}g carbs · {item.fat}g fat ·{" "}
                    {item.fiber}g fiber
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