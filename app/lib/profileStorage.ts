import { profile as defaultProfile } from "@/app/data/profile";
import type { UserProfile } from "@/app/types/profile";

const STORAGE_KEY = "userProfile";

export function getStoredProfile(): UserProfile {
  if (typeof window === "undefined") {
    return defaultProfile;
  }

  const savedProfile =
    localStorage.getItem(STORAGE_KEY);

  if (!savedProfile) {
    return defaultProfile;
  }

  try {
    return JSON.parse(savedProfile) as UserProfile;
  } catch {
    return defaultProfile;
  }
}

export function saveStoredProfile(
  profile: UserProfile
) {
  if (typeof window === "undefined") {
    return;
  }

  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(profile)
  );
}