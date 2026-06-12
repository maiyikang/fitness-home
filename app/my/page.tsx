"use client";

import { useState } from "react";

export default function MyPage() {
  const [bodyFat, setBodyFat] = useState("");
  const [proteinTarget, setProteinTarget] = useState("");
  const [fiberTarget, setFiberTarget] = useState("");
  const [carbTarget, setCarbTarget] = useState("");
  const [fatTarget, setFatTarget] = useState("");
  const [favoriteFoods, setFavoriteFoods] = useState("");

  return (
    <main className="min-h-screen bg-gray-50 px-6 py-10">
      <section className="mx-auto max-w-3xl rounded-2xl bg-white p-8 shadow-sm">
        <h1 className="text-3xl font-bold text-gray-900">
          My Profile
        </h1>

        <p className="mt-3 text-gray-600">
          Manage your nutrition preferences.
        </p>

        <div className="mt-8 grid gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Body Fat %
            </label>

            <input
              type="number"
              value={bodyFat}
              onChange={(e) => setBodyFat(e.target.value)}
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Daily Protein Target (g)
            </label>

            <input
              type="number"
              value={proteinTarget}
              onChange={(e) => setProteinTarget(e.target.value)}
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Daily Fiber Target (g)
            </label>

            <input
              type="number"
              value={fiberTarget}
              onChange={(e) => setFiberTarget(e.target.value)}
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Daily Carb Target (g)
            </label>

            <input
              type="number"
              value={carbTarget}
              onChange={(e) => setCarbTarget(e.target.value)}
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Daily Fat Target (g)
            </label>

            <input
              type="number"
              value={fatTarget}
              onChange={(e) => setFatTarget(e.target.value)}
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Favorite Foods
            </label>

            <input
              type="text"
              value={favoriteFoods}
              onChange={(e) => setFavoriteFoods(e.target.value)}
              placeholder="Sushi, Chicken, Steak..."
              className="mt-2 w-full rounded-lg border border-gray-300 px-4 py-3"
            />
          </div>

          <button
            className="rounded-lg bg-black px-6 py-3 font-medium text-white"
          >
            Save Preferences
          </button>
        </div>
      </section>
    </main>
  );
}
