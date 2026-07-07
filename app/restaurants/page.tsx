"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import BottomNavigation from "@/app/components/BottomNavigation";
import FavoriteButton from "@/app/components/FavoriteButton";
import { restaurants as sampleRestaurants } from "@/app/data/restaurants";
import { profile as defaultProfile } from "@/app/data/profile";
import { getStoredProfile } from "@/app/lib/profileStorage";
import { saveRealRestaurants } from "@/app/lib/realRestaurantsStorage";
import {
  calculateMatchScore,
  getMatchScoreReasons,
} from "@/app/lib/scoring";
import type { UserProfile } from "@/app/types/profile";
import type { Restaurant } from "@/app/types/restaurant";

type AiRetrievedResult = {
  rank: number;
  restaurant_name: string;
  category: string;
  similarity_score: number;
  average_calories: string;
  average_protein: string;
  average_fiber: string;
};

type AiRecommendationResult = {
  query: string;
  retrieved_count: number;
  retrieved_results: AiRetrievedResult[];
  generated_answer: string;
};

export default function RestaurantsPage() {
  const [userProfile, setUserProfile] =
    useState<UserProfile>(defaultProfile);

  const [restaurantList, setRestaurantList] =
    useState<Restaurant[]>(sampleRestaurants);

  const [isLoadingPlaces, setIsLoadingPlaces] = useState(false);
  const [placesMessage, setPlacesMessage] = useState("");

  const [aiQuery, setAiQuery] = useState(
    "I want a high-protein Japanese meal under 600 calories."
  );
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [aiError, setAiError] = useState("");
  const [aiResult, setAiResult] =
    useState<AiRecommendationResult | null>(null);

  useEffect(() => {
    setUserProfile(getStoredProfile());
  }, []);

  async function handleLoadRealRestaurants() {
    setIsLoadingPlaces(true);
    setPlacesMessage("");

    try {
      const response = await fetch("/api/places/search");

      if (!response.ok) {
        setPlacesMessage("Failed to load real restaurants.");
        return;
      }

      const realRestaurants = (await response.json()) as Restaurant[];

      if (realRestaurants.length === 0) {
        setPlacesMessage("No restaurants found from Google Places.");
        return;
      }

      saveRealRestaurants(realRestaurants);
      setRestaurantList(realRestaurants);
      setPlacesMessage("Real restaurants loaded from Google Places.");
    } catch {
      setPlacesMessage("Something went wrong while loading restaurants.");
    } finally {
      setIsLoadingPlaces(false);
    }
  }

  function handleUseSampleRestaurants() {
    setRestaurantList(sampleRestaurants);
    setPlacesMessage("Sample restaurants restored.");
  }

  async function handleAiRecommend() {
    setIsAiLoading(true);
    setAiError("");
    setAiResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: aiQuery,
          top_k: 5,
        }),
      });

      if (!response.ok) {
        setAiError("Failed to get AI recommendation.");
        return;
      }

      const data = (await response.json()) as AiRecommendationResult;
      setAiResult(data);
    } catch {
      setAiError("Backend is not running. Please start FastAPI first.");
    } finally {
      setIsAiLoading(false);
    }
  }

  const sortedRestaurants = [...restaurantList].sort((a, b) => {
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

          <div className="mt-5 rounded-3xl bg-white p-5 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold text-gray-900">
                  AI Fitness Meal Recommendation
                </h2>

                <p className="mt-2 text-sm text-gray-500">
                  Ask the RAG + Llama backend for a personalized restaurant recommendation.
                </p>
              </div>

              <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-semibold text-green-700">
                RAG + Llama
              </span>
            </div>

            <textarea
              value={aiQuery}
              onChange={(event) => setAiQuery(event.target.value)}
              className="mt-4 h-28 w-full rounded-2xl border border-gray-200 p-4 text-sm outline-none focus:border-green-500"
              placeholder="Example: I want a high-protein Japanese meal under 600 calories."
            />

            <button
              type="button"
              onClick={handleAiRecommend}
              disabled={isAiLoading || aiQuery.trim().length === 0}
              className="mt-4 rounded-full bg-black px-5 py-3 text-sm font-semibold text-white disabled:opacity-50"
            >
              {isAiLoading ? "Generating..." : "Get AI Recommendation"}
            </button>

            {aiError && (
              <p className="mt-3 text-sm font-medium text-red-600">
                {aiError}
              </p>
            )}

            {aiResult && (
              <div className="mt-5 space-y-4">
                <div className="rounded-2xl bg-green-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-green-700">
                    AI Recommendation
                  </p>

                  <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-green-950">
                    {aiResult.generated_answer}
                  </p>
                </div>

                <div className="rounded-2xl bg-gray-50 p-4">
                  <p className="text-sm font-bold text-gray-900">
                    Retrieved Evidence
                  </p>

                  <p className="mt-1 text-xs text-gray-500">
                    Query: {aiResult.query}
                  </p>

                  <div className="mt-3 space-y-3">
                    {aiResult.retrieved_results.slice(0, 3).map((result) => (
                      <div
                        key={`${result.rank}-${result.restaurant_name}`}
                        className="rounded-2xl bg-white p-3 text-sm shadow-sm"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="font-bold text-gray-900">
                              #{result.rank} {result.restaurant_name}
                            </p>

                            <p className="mt-1 text-gray-500">
                              {result.category}
                            </p>
                          </div>

                          <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-semibold text-green-700">
                            {result.similarity_score.toFixed(3)}
                          </span>
                        </div>

                        <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                          <div className="rounded-xl bg-gray-50 p-2">
                            <p className="text-xs text-gray-400">
                              Calories
                            </p>
                            <p className="text-sm font-bold text-gray-900">
                              {result.average_calories}
                            </p>
                          </div>

                          <div className="rounded-xl bg-gray-50 p-2">
                            <p className="text-xs text-gray-400">
                              Protein
                            </p>
                            <p className="text-sm font-bold text-gray-900">
                              {result.average_protein}
                            </p>
                          </div>

                          <div className="rounded-xl bg-gray-50 p-2">
                            <p className="text-xs text-gray-400">
                              Fiber
                            </p>
                            <p className="text-sm font-bold text-gray-900">
                              {result.average_fiber}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="mt-5 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={handleLoadRealRestaurants}
              disabled={isLoadingPlaces}
              className="rounded-full bg-black px-5 py-3 text-sm font-semibold text-white disabled:opacity-50"
            >
              {isLoadingPlaces ? "Loading..." : "Load Real Restaurants"}
            </button>

            <button
              type="button"
              onClick={handleUseSampleRestaurants}
              className="rounded-full bg-white px-5 py-3 text-sm font-semibold text-gray-700 shadow-sm"
            >
              Use Sample Data
            </button>
          </div>

          {placesMessage && (
            <p className="mt-3 text-sm font-medium text-green-700">
              {placesMessage}
            </p>
          )}
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
                href={`/restaurants/${encodeURIComponent(restaurant.id)}`}
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

                    <p className="mt-1 text-xs font-medium text-gray-400">
                      {restaurant.cuisine}
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