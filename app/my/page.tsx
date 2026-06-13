"use client";

import { useEffect, useState } from "react";

import BottomNavigation from "@/app/components/BottomNavigation";
import { profile as defaultProfile } from "@/app/data/profile";
import {
  calculateBMI,
  calculateBMR,
  calculateNutritionTargets,
  calculateTDEE,
} from "@/app/lib/nutrition";
import {
  getStoredProfile,
  saveStoredProfile,
} from "@/app/lib/profileStorage";
import type { UserProfile } from "@/app/types/profile";

export default function MyPage() {
  const [profile, setProfile] = useState<UserProfile>(defaultProfile);
  const [savedMessage, setSavedMessage] = useState("");

  useEffect(() => {
    const storedProfile = getStoredProfile();
    setProfile(storedProfile);
  }, []);

  function handleSave() {
    saveStoredProfile(profile);

    window.dispatchEvent(new Event("profileUpdated"));

    setSavedMessage("Profile saved successfully.");
  }

  const bmi = calculateBMI(profile.height, profile.weight);
  const bmr = calculateBMR(profile);
  const tdee = calculateTDEE(profile);
  const nutritionTargets = calculateNutritionTargets(profile);

  return (
    <main className="min-h-screen bg-gray-50 px-6 pb-24 pt-10">
      <section className="mx-auto max-w-3xl rounded-2xl bg-white p-8 shadow-sm">
        <h1 className="text-3xl font-bold text-gray-900">My Profile</h1>

        <p className="mt-3 text-gray-600">
          Manage your body data, meal preferences, and nutrition targets.
        </p>

        <div className="mt-8">
          <h2 className="text-xl font-bold text-gray-900">Body Summary</h2>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl bg-gray-50 p-4">
              <p className="text-sm text-gray-500">BMI</p>
              <p className="mt-1 text-xl font-bold text-gray-900">{bmi}</p>
            </div>

            <div className="rounded-2xl bg-gray-50 p-4">
              <p className="text-sm text-gray-500">BMR</p>
              <p className="mt-1 text-xl font-bold text-gray-900">
                {bmr} kcal
              </p>
            </div>

            <div className="rounded-2xl bg-gray-50 p-4">
              <p className="text-sm text-gray-500">TDEE</p>
              <p className="mt-1 text-xl font-bold text-gray-900">
                {tdee} kcal
              </p>
            </div>

            <div className="rounded-2xl bg-gray-50 p-4">
              <p className="text-sm text-gray-500">Target Calories</p>
              <p className="mt-1 text-xl font-bold text-gray-900">
                {nutritionTargets.calories} kcal
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8">
          <h2 className="text-xl font-bold text-gray-900">
            Daily Nutrition Targets
          </h2>

          <p className="mt-2 text-sm text-gray-500">
            These targets will be used later to recommend better meals.
          </p>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl bg-green-50 p-4">
              <p className="text-sm text-green-700">Calories</p>
              <p className="mt-1 text-2xl font-bold text-green-800">
                {nutritionTargets.calories} kcal
              </p>
            </div>

            <div className="rounded-2xl bg-green-50 p-4">
              <p className="text-sm text-green-700">Protein</p>
              <p className="mt-1 text-2xl font-bold text-green-800">
                {nutritionTargets.protein} g
              </p>
            </div>

            <div className="rounded-2xl bg-green-50 p-4">
              <p className="text-sm text-green-700">Carbs</p>
              <p className="mt-1 text-2xl font-bold text-green-800">
                {nutritionTargets.carbs} g
              </p>
            </div>

            <div className="rounded-2xl bg-green-50 p-4">
              <p className="text-sm text-green-700">Fat</p>
              <p className="mt-1 text-2xl font-bold text-green-800">
                {nutritionTargets.fat} g
              </p>
            </div>

            <div className="rounded-2xl bg-green-50 p-4 md:col-span-2">
              <p className="text-sm text-green-700">Fiber</p>
              <p className="mt-1 text-2xl font-bold text-green-800">
                {nutritionTargets.fiber} g
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8 grid gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Height cm
            </label>
            <input
              type="number"
              value={profile.height}
              onChange={(event) =>
                setProfile({
                  ...profile,
                  height: Number(event.target.value),
                })
              }
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Weight kg
            </label>
            <input
              type="number"
              value={profile.weight}
              onChange={(event) =>
                setProfile({
                  ...profile,
                  weight: Number(event.target.value),
                })
              }
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Age
            </label>
            <input
              type="number"
              value={profile.age}
              onChange={(event) =>
                setProfile({
                  ...profile,
                  age: Number(event.target.value),
                })
              }
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Gender
            </label>
            <select
              value={profile.gender}
              onChange={(event) =>
                setProfile({
                  ...profile,
                  gender: event.target.value as UserProfile["gender"],
                })
              }
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            >
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Activity Level
            </label>
            <select
              value={profile.activityLevel}
              onChange={(event) =>
                setProfile({
                  ...profile,
                  activityLevel:
                    event.target.value as UserProfile["activityLevel"],
                })
              }
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Goal
            </label>
            <select
              value={profile.goal}
              onChange={(event) =>
                setProfile({
                  ...profile,
                  goal: event.target.value as UserProfile["goal"],
                })
              }
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            >
              <option value="lose-fat">Lose Fat</option>
              <option value="maintain">Maintain</option>
              <option value="gain-muscle">Gain Muscle</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Meal Type
            </label>
            <select
              value={profile.mealType}
              onChange={(event) =>
                setProfile({
                  ...profile,
                  mealType: event.target.value as UserProfile["mealType"],
                })
              }
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            >
              <option value="healthy">Healthy</option>
              <option value="normal">Normal</option>
              <option value="cheat">Cheat</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Favorite Cuisine
            </label>
            <input
              type="text"
              value={profile.favoriteCuisine.join(", ")}
              onChange={(event) =>
                setProfile({
                  ...profile,
                  favoriteCuisine: event.target.value
                    .split(",")
                    .map((item) => item.trim())
                    .filter(Boolean),
                })
              }
              placeholder="Healthy, Sushi, Burger"
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <button
            type="button"
            onClick={handleSave}
            className="rounded-lg bg-black px-6 py-3 font-medium text-white"
          >
            Save Profile
          </button>

          {savedMessage && (
            <p className="text-sm font-medium text-green-600">
              {savedMessage}
            </p>
          )}
        </div>
      </section>

      <BottomNavigation />
    </main>
  );
}