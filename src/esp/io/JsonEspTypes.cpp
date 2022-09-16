// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "JsonAllTypes.h"

namespace esp {
namespace io {

JsonGenericValue toJsonValue(const gfx::replay::Keyframe& keyframe,
                             JsonAllocator& allocator) {
  JsonGenericValue obj(rapidjson::kObjectType);

  io::addMember(obj, "loads", keyframe.loads, allocator);

  if (!keyframe.creations.empty()) {
    JsonGenericValue creationsArray(rapidjson::kArrayType);
    for (const auto& pair : keyframe.creations) {
      JsonGenericValue creationPairObj(rapidjson::kObjectType);
      io::addMember(creationPairObj, "instanceKey", pair.first, allocator);
      io::addMember(creationPairObj, "creation", pair.second, allocator);

      creationsArray.PushBack(creationPairObj, allocator);
    }
    io::addMember(obj, "creations", creationsArray, allocator);
  }

  io::addMember(obj, "deletions", keyframe.deletions, allocator);

  if (!keyframe.stateUpdates.empty()) {
    JsonGenericValue stateUpdatesArray(rapidjson::kArrayType);
    for (const auto& pair : keyframe.stateUpdates) {
      JsonGenericValue stateObj(rapidjson::kObjectType);
      io::addMember(stateObj, "instanceKey", pair.first, allocator);
      io::addMember(stateObj, "state", pair.second, allocator);
      stateUpdatesArray.PushBack(stateObj, allocator);
    }
    io::addMember(obj, "stateUpdates", stateUpdatesArray, allocator);
  }

  if (!keyframe.userTransforms.empty()) {
    JsonGenericValue userTransformsArray(rapidjson::kArrayType);
    for (const auto& pair : keyframe.userTransforms) {
      JsonGenericValue wrapperObj(rapidjson::kObjectType);
      io::addMember(wrapperObj, "name", pair.first, allocator);
      io::addMember(wrapperObj, "transform", pair.second, allocator);
      userTransformsArray.PushBack(wrapperObj, allocator);
    }
    io::addMember(obj, "userTransforms", userTransformsArray, allocator);
  }

  if (!keyframe.lights.empty()) {
    JsonGenericValue lightsArray(rapidjson::kArrayType);
    for (const auto& light : keyframe.lights) {
      JsonGenericValue lightObj(rapidjson::kObjectType);
      io::addMember(lightObj, "light", light, allocator);
      lightsArray.PushBack(lightObj, allocator);
    }
    io::addMember(obj, "lights", lightsArray, allocator);
  }

  return obj;
}

bool fromJsonValue(const JsonGenericValue& obj,
                   gfx::replay::Keyframe& keyframe) {
  io::readMember(obj, "loads", keyframe.loads);

  auto itr = obj.FindMember("creations");
  if (itr != obj.MemberEnd()) {
    const JsonGenericValue& creationsArray = itr->value;
    keyframe.creations.reserve(creationsArray.Size());
    for (const auto& creationPairObj : creationsArray.GetArray()) {
      std::pair<gfx::replay::RenderAssetInstanceKey,
                assets::RenderAssetInstanceCreationInfo>
          pair;
      io::readMember(creationPairObj, "instanceKey", pair.first);
      io::readMember(creationPairObj, "creation", pair.second);
      keyframe.creations.emplace_back(std::move(pair));
    }
  }

  io::readMember(obj, "deletions", keyframe.deletions);

  itr = obj.FindMember("stateUpdates");
  if (itr != obj.MemberEnd()) {
    const JsonGenericValue& stateUpdatesArray = itr->value;
    keyframe.stateUpdates.reserve(stateUpdatesArray.Size());
    for (const auto& stateObj : stateUpdatesArray.GetArray()) {
      std::pair<gfx::replay::RenderAssetInstanceKey,
                gfx::replay::RenderAssetInstanceState>
          pair;
      io::readMember(stateObj, "instanceKey", pair.first);
      io::readMember(stateObj, "state", pair.second);
      keyframe.stateUpdates.emplace_back(std::move(pair));
    }
  }

  itr = obj.FindMember("userTransforms");
  if (itr != obj.MemberEnd()) {
    const JsonGenericValue& userTransformsArray = itr->value;
    for (const auto& userTransformObj : userTransformsArray.GetArray()) {
      std::string name;
      gfx::replay::Transform transform;
      io::readMember(userTransformObj, "name", name);
      io::readMember(userTransformObj, "transform", transform);
      keyframe.userTransforms[name] = transform;
    }
  }

  itr = obj.FindMember("lights");
  if (itr != obj.MemberEnd()) {
    const JsonGenericValue& lightsArray = itr->value;
    for (const auto& lightObj : lightsArray.GetArray()) {
      gfx::LightInfo light;
      io::readMember(lightObj, "light", light);
      keyframe.lights.emplace_back(std::move(light));
    }
  }

  return true;
}

JsonGenericValue toJsonValue(const esp::assets::AssetInfo& x,
                             JsonAllocator& allocator) {
  JsonGenericValue obj(rapidjson::kObjectType);
  addMemberAsUint32(obj, "type", x.type, allocator);
  addMember(obj, "filepath", x.filepath, allocator);
  addMember(obj, "frame", x.frame, allocator);
  addMember(obj, "virtualUnitToMeters", x.virtualUnitToMeters, allocator);
  addMember(obj, "forceFlatShading", x.forceFlatShading, allocator);
  addMember(obj, "splitInstanceMesh", x.splitInstanceMesh, allocator);
  addMember(obj, "shaderTypeToUse", x.shaderTypeToUse, allocator);
  addMember(obj, "overridePhongMaterial", x.overridePhongMaterial, allocator);
  addMember(obj, "hasSemanticTextures", x.hasSemanticTextures, allocator);

  return obj;
}

bool fromJsonValue(const JsonGenericValue& obj, esp::assets::AssetInfo& x) {
  readMemberAsUint32(obj, "type", x.type);
  readMember(obj, "filepath", x.filepath);
  readMember(obj, "frame", x.frame);
  readMember(obj, "virtualUnitToMeters", x.virtualUnitToMeters);
  readMember(obj, "forceFlatShading", x.forceFlatShading);
  readMember(obj, "splitInstanceMesh", x.splitInstanceMesh);
  readMember(obj, "shaderTypeToUse", x.shaderTypeToUse);
  readMember(obj, "overridePhongMaterial", x.overridePhongMaterial);
  readMember(obj, "hasSemanticTextures", x.hasSemanticTextures);
  return true;
}

JsonGenericValue toJsonValue(
    const metadata::attributes::ObjectInstanceShaderType& x,
    JsonAllocator& allocator) {
  return toJsonValue(metadata::attributes::getShaderTypeName(x), allocator);
}

bool fromJsonValue(const JsonGenericValue& obj,
                   metadata::attributes::ObjectInstanceShaderType& x) {
  std::string shaderTypeToUseString;
  // read as string
  bool shaderTypeSucceess = fromJsonValue(obj, shaderTypeToUseString);
  // convert to enum
  if (shaderTypeSucceess) {
    const std::string shaderTypeLC =
        Cr::Utility::String::lowercase(shaderTypeToUseString);
    auto mapIter = metadata::attributes::ShaderTypeNamesMap.find(shaderTypeLC);
    ESP_CHECK(mapIter != metadata::attributes::ShaderTypeNamesMap.end(),
              "Illegal shader_type value '"
                  << shaderTypeToUseString
                  << "' specified in JSON to be used to set "
                     "AssetInfo.shaderTypeToUse. "
                     "Aborting.");
    x = mapIter->second;
  }
  return shaderTypeSucceess;
}

JsonGenericValue toJsonValue(const esp::gfx::LightPositionModel& x,
                             JsonAllocator& allocator) {
  return toJsonValue(metadata::attributes::getLightPositionModelName(x),
                     allocator);
}

bool fromJsonValue(const JsonGenericValue& obj,
                   esp::gfx::LightPositionModel& x) {
  std::string lightPositionModelString;
  // read as string
  bool success = fromJsonValue(obj, lightPositionModelString);
  // convert to enum
  if (success) {
    const std::string lightPositionModelLC =
        Cr::Utility::String::lowercase(lightPositionModelString);
    auto mapIter =
        metadata::attributes::LightPositionNamesMap.find(lightPositionModelLC);
    ESP_CHECK(mapIter != metadata::attributes::LightPositionNamesMap.end(),
              "Illegal model value '"
                  << lightPositionModelString
                  << "' specified in JSON to be used to set LightInfo.model. "
                  << "Aborting.");
    x = mapIter->second;
  }
  return success;
}

}  // namespace io
}  // namespace esp
